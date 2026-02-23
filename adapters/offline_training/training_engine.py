"""
Offline Training Engine (Adapter Layer)

Responsible for:
- Training ML models (state estimator, predictor) on historical data
- Exporting trained model weights
- Pushing updates to the production system via IModelUpdatePort
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import pickle
from typing import TYPE_CHECKING, Any

from core.ml.services import IModelUpdatePort

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from core.domain.models import Instrument


@dataclass(slots=True)
class TrainedModelArtifact:
    version: str
    weights: dict[str, Any]
    metadata: dict[str, Any]


class TrainingEngine:
    """
    Offline training adapter for model development and refinement.
    
    ⚠️ IMPORTANT: This is an OFFLINE training adapter, NOT the production predictor.
    The production predictor (MambaPredictor) receives trained weights via push_model()
    and applies them via apply_pending_update().
    
    If flat equity persists after training, check:
    1. MambaPredictor.apply_pending_update() - does it apply both predictor AND state_estimator?
    2. StateEstimator - is it using the updated weights from training?
    3. Prediction pipeline - is it actually calling the updated models?
    
    Known ports (imports):
    - IModelUpdatePort: for pushing trained weights to production predictor
    
    Typical workflow:
    1. Load historical dataset
    2. train() - iterate over data, accumulate gradients
    3. export_weights() - serialize model for distribution
    4. push_model() - send to production via ModelUpdatePort
    """
    
    def __init__(self) -> None:
        """Initialize training engine with empty state."""
        self._artifact: TrainedModelArtifact | None = None
        self.training_history: list[dict[str, Any]] = []
        self.trained_version: str = "untrained"

    @staticmethod
    def _extract_returns(dataset: list[dict[str, Any]]) -> list[float]:
        returns: list[float] = []
        prev_price: float | None = None
        for row in dataset:
            if "return" in row:
                returns.append(float(row["return"]))
                continue
            price = float(row.get("price", 0.0))
            if prev_price is None or prev_price == 0.0:
                returns.append(0.0)
            else:
                returns.append((price - prev_price) / prev_price)
            prev_price = price
        return returns

    def train(
        self,
        dataset: list[dict[str, Any]],
        instrument: Instrument | None = None,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        Train predictor + state_estimator on historical dataset (OFFLINE).
        
        Args:
            dataset: List of dicts with keys like:
                    - 'price': float
                    - 'return': float
                    - 'volume': float
                    - 'ts': datetime
            instrument: Target instrument (optional)
                       For multi-symbol training, specify instrument explicitly
                       For single-symbol or generic models, None is acceptable
            epochs: Number of training iterations over the dataset
            batch_size: Mini-batch size (validated, used for batch count)
        
        Returns:
            dict with training metadata:
            - 'epochs': int
            - 'samples': int
            - 'batches': int
            - 'loss': float (mock)
            - 'version': str (unique UTC timestamp)
        """
        if not dataset:
            raise ValueError("Dataset cannot be empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        returns = self._extract_returns(dataset)
        if len(returns) < 3:
            raise ValueError("Dataset too short: need at least 3 samples for training")

        total_samples = len(dataset) * epochs
        num_batches = (len(dataset) + batch_size - 1) // batch_size

        # Simple supervised objective: predict next return from current return, delta-return, abs-return
        x_rows: list[list[float]] = []
        y_rows: list[list[float]] = []
        for idx in range(1, len(returns) - 1):
            r_t = float(returns[idx])
            r_prev = float(returns[idx - 1])
            r_next = float(returns[idx + 1])
            x_rows.append([r_t, r_t - r_prev, abs(r_t), 1.0])
            y_rows.append([r_next])

        if not x_rows:
            raise ValueError("Failed to build training windows from dataset")

        # Torch-accelerated linear fit when available, otherwise closed-form fallback
        if torch is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            x = torch.tensor(x_rows, dtype=torch.float32, device=device)
            y = torch.tensor(y_rows, dtype=torch.float32, device=device)
            model = torch.nn.Linear(4, 1, bias=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            last_loss = 0.0
            for _ in range(max(1, epochs * 25)):
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu().item())

            with torch.no_grad():
                w = model.weight.detach().cpu().view(-1).tolist()
                b = float(model.bias.detach().cpu().item())
            train_device = device
            train_loss = last_loss
        else:
            # Lightweight non-torch fallback: mean-reversion coefficient estimate
            # mu_{t+1} ≈ a * r_t + b
            xs = [row[0] for row in x_rows]
            ys = [row[0] for row in y_rows]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            num = sum((a - mean_x) * (b_ - mean_y) for a, b_ in zip(xs, ys))
            den = sum((a - mean_x) ** 2 for a in xs) or 1e-12
            a = num / den
            b = mean_y - a * mean_x
            w = [a, 0.0, 0.0, 0.0]
            train_device = "cpu"
            train_loss = sum((a * x_i + b - y_i) ** 2 for x_i, y_i in zip(xs, ys)) / max(1, len(xs))

        vol = max(1e-6, (sum(r * r for r in returns) / len(returns)) ** 0.5)

        # Explicit separation: predictor + state_estimator weights
        weights = {
            "predictor": {
                "head": {
                    "w": [float(v) for v in w],
                    "b": float(b),
                },
                "mamba": {
                    "d_model": 4,
                    "n_state": 8,
                    "delta_scale": 0.1,
                    "A": [[-0.1 - 0.01 * j for j in range(8)] for _ in range(4)],
                    "B_base": [0.1 for _ in range(8)],
                    "C_base": [0.1 for _ in range(8)],
                    "D": [0.0 for _ in range(4)],
                },
                "volatility": float(vol),
                "hyperparams": {"device": train_device, "epochs": epochs},
            },
            "state_estimator": {
                "state_dict": {"process_noise": float(vol), "measurement_noise": float(vol * 2.0)},
                "hyperparams": {
                    "state_dim": 16,
                    "device": train_device,
                },
            },
        }
        
        # Unique version string with UTC timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.trained_version = f"offline-{timestamp}-e{epochs}"
        
        result: dict[str, Any] = {
            "epochs": epochs,
            "samples": total_samples,
            "batches": num_batches,
            "loss": float(train_loss),
            "version": self.trained_version,
            "device": train_device,
        }

        self._artifact = TrainedModelArtifact(
            version=self.trained_version,
            weights=weights,
            metadata={
                "epochs": epochs,
                "batch_size": batch_size,
                "batches_per_epoch": num_batches,
                "samples": total_samples,
                "instrument": getattr(instrument, "symbol", None) if instrument else None,
            },
        )
        
        self.training_history.append(result)
        return result

    def export_weights(self) -> bytes:
        """
        Serialize model weights for distribution.
        
        Returns:
            bytes in format:
            - First 8 bytes: version string length (big-endian uint64)
            - Next N bytes: version string (UTF-8)
            - Remaining: pickled model_weights dict
        
        This format is consumed by MambaPredictor.request_model_update()
        
        ⚠️ Production Risk: pickle is not secure and not language-agnostic.
        For production systems, consider:
        - torch.save() for PyTorch models
        - joblib for scikit-learn
        - msgpack + numpy for cross-language compatibility
        - protobuf for strict versioning
        """
        if self._artifact is None:
            raise RuntimeError("No trained weights available")
        
        # Encode version
        version_bytes = self._artifact.version.encode("utf-8")
        version_length = len(version_bytes).to_bytes(8, byteorder="big")
        
        # Pickle model weights (MVP only - see docstring for production alternatives)
        weights_payload = pickle.dumps(self._artifact.weights)
        
        # Combine
        return version_length + version_bytes + weights_payload

    async def push_model(self, update_port: IModelUpdatePort) -> None:
        """
        Push trained model to production via ModelUpdatePort.
        
        The production system (TradingEngine -> MambaPredictor) will:
        1. Receive update via request_model_update()
        2. Store in staging model
        3. Apply when convenient (apply_pending_update())
        
        Args:
            update_port: IModelUpdatePort implementer (e.g., MambaPredictor)
        """
        if self._artifact is None:
            raise RuntimeError("No trained weights available")
        weights_bytes = self.export_weights()
        await update_port.request_model_update(weights_bytes, self._artifact.version)

    def get_training_history(self) -> list[dict[str, Any]]:
        """Return list of training runs."""
        return list(self.training_history)

    def get_latest_artifact(self) -> TrainedModelArtifact | None:
        """Return latest trained model artifact, if available."""
        return self._artifact

