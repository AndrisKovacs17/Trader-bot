# KCA-Mamba kereskedési rendszer (felhasználói útmutató)

Ez a dokumentum a [diplomamunka](diplomamunka/main.pdf) kiegészítő felhasználói
leírása. A fejlesztői dokumentáció a dolgozatban (4–5. fejezet) található.

## 1. Mire való a szoftver?

A program egy kriptovaluta-kereskedési rendszer demóalkalmazása, amely a
Binance tőzsde nyilvános adatait használja. A rendszer:

* Valós idejű piaci adatokat vesz át a Binance WebSocket API-n keresztül,
* Egy saját fejlesztésű Kalman-szűrővel bővített Mamba (KCA-Mamba) modellel
  rövidtávú ár-előrejelzést készít,
* Kockázati szabályokkal szűrt, paper trading üzemmódú kereskedési
  szignálokat generál,
* Webes irányítópulton (dashboard) megjeleníti a modell állapotát, a
  szignálokat, az egyenleget és a diagnosztikákat.

A program **nem** éles kereskedésre készült. A pénzügyi veszteségek
elkerülése érdekében minden order a beépített szimulációs brókerbe fut.

## 2. Célközönség

| Felhasználó típus | Tipikus feladat |
|---|---|
| Kutató / hallgató | A KCMamba modell teljesítményének reprodukciója a `tests/` suite-on keresztül |
| Fejlesztő | A hexagonális architektúra bővítése új adapterekkel |
| Szakmai bíráló | A rendszer működésének megtekintése a dashboardon |

## 3. Minimális rendszerkövetelmény

| Komponens | Minimum | Javasolt |
|---|---|---|
| Operációs rendszer | Linux (Ubuntu 22.04+) / Windows 10+ / macOS 13+ | Linux |
| Python | 3.11 | 3.12 |
| CPU | 4 mag, x86-64 | 8 mag |
| RAM | 4 GB | 8 GB |
| GPU | nincs (CPU-n is fut) | NVIDIA CUDA 12.x kompatibilis |
| Lemez | 2 GB (dataset cache-sel együtt ~150 MB) | 5 GB |
| Hálózat | Binance REST/WS (kimenő HTTPS, 443) | stabil szélessáv |

## 4. Telepítés

```bash
# 1. repository klónozása
git clone https://github.com/AndrisKovacs17/Trader-bot.git
cd Trader-bot

# 2. virtuális környezet létrehozása
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. függőségek telepítése
pip install -r requirements.txt     # futtatáshoz
pip install -r requirements-ml.txt  # a benchmark/tréning scriptekhez
```

A `torch` GPU-s változatához az NVIDIA által ajánlott wheelt kell használni
(<https://pytorch.org/get-started/>). CPU-n is minden futtatható, csak a
benchmark lassabb.

## 5. A program indítása

### 5.1 Fő demó (live + dashboard)

```bash
python main.py
```

Ez elindít:

* egy háttérben futó tréningfolyamatot (offline dataset alapján),
* a Binance WebSocket élő adatfolyamot,
* a webes dashboardot a `http://127.0.0.1:8080/` címen.

Leállítás: **Ctrl+C**. A program ilyenkor szabályosan lezárja a WebSocketet,
leállítja a HTTP-szervert és kiírja a végleges pozíció-összegzést.

### 5.2 Tesztek futtatása

```bash
python -m pytest tests/ -q
```

Várt kimenet: `71 passed`.

## 6. Dashboard (felhasználói felület)

A dashboard magyar nyelvű, a felső navigációs sávban kategorizált menükkel:

| Menü | Oldal | Tartalom |
|---|---|---|
| Pénzügyi | Áttekintés (`/`) | Egyenleg, P&L, aktív pozíciók, élő chart |
| Pénzügyi | Modell (`/model.html`) | Modellverzió, kalibráció, diagnosztikai metrikák |
| Pénzügyi | Jelek (`/signals.html`) | Generált szignálok listája, kockázati státusszal |
| Pénzügyi | Tréning (`/training.html`) | Offline tréningfolyamat előrehaladása |
| Benchmark | Szintetikus teszt (`/benchmark.html`) | Kontrollált zajszintű szintetikus teszt eredménye |
| Benchmark | LSTM / Mamba összehasonlítás (`/ltsf_benchmark.html`) | LTSF adaton mért eredmények |
| KCA elemzés | Kalman-szűrő diagnosztika (`/kla_kalman.html`) | K, A, R mátrixok időbeli alakulása |
| KCA elemzés | Modell komplexitás térkép (`/complexity.html`) | Paraméterszám × teljesítmény |

**Súgó a dashboardon**: a jobb alsó sarokban megjelenő `?` gombra kattintva
oldal-specifikus súgóablak nyílik, amely elmagyarázza az adott képernyő
funkcióit.

## 7. Tipikus hibaüzenetek és teendők

| Üzenet | Ok | Teendő |
|---|---|---|
| `[WARN] RSS/sentiment processing failed: ...` | RSS forrás átmenetileg elérhetetlen | Figyelmen kívül hagyható, a program hír nélkül folytatja. |
| `[FALLBACK] WebSocket unavailable. Switching to REST polling` | A WebSocket kapcsolat instabil | A program automatikusan áttér REST-polling fallback módra, manuális beavatkozás nélkül. |
| `[FATAL ERROR] ...` + traceback | Váratlan belső hiba | A program lezárja a kapcsolatokat és REST fallbackre vált. Ismétlődő jelentkezés esetén érdemes hibajegyet nyitni. |
| `[SHUTDOWN] User interrupted (CTRL+C)` | Szabályos megszakítás | Nincs teendő, a `finally` ág elvégzi az erőforrás-felszabadítást. |
| `Bad request: API key required` | Érvénytelen Binance kulcs | Historikus és WebSocket adatokhoz kulcs nem szükséges, és a rendszer paper trading üzemmódja miatt éles rendeléshez sincs rá szükség. |
| `OSError: [Errno 98] Address already in use` | A 8080-as port foglalt | Állítsa le a foglaló folyamatot, vagy módosítsa a `config.web.port` értéket |

A program minden hibaüzenetet a `stderr`-re ír, a rendes kimenet a `stdout`-ra
megy, így `nohup python main.py > run.log 2> run.err &` módon külön
naplózható.

## 8. Biztonsági megfontolások

* A rendszer paper trading módban fut, és nem nyit valós tőzsdei pozíciót.
* A Binance kulcsok (ha mégis használ) csak olvasási jogosultságúak legyenek.
* A dashboard csak a lokális interfészen (`127.0.0.1`) hallgat, nem nyilvános.
* A tréning cache (`adapters/offline_training/dataset_cache/`) nincs a
  repositoryban (lásd `.gitignore`).

## 9. Megszakíthatóság

A program hosszú műveletei (tréning, benchmark sweep, live stream) **Ctrl+C**-vel
bármikor leállíthatók **Ctrl+C**-vel. Leállításkor:

* a WebSocket kapcsolat lezárul,
* az esemény-pufferek tartalma kiürül,
* a HTTP-szerver lezárja az aktív kapcsolatokat,
* a végleges állapot megjelenik a konzolon.

## 10. A KCA-Mamba modell architektúrája

### 10.1 Motiváció

A hagyományos Mamba SSM (State Space Model) szelektív kapui a szekvencia
tartalmától függően változtatják az állapotátmeneti mátrixot, de a zajszint
becslése implicit marad: a modellnek magának kell megtanulnia, mikor bízzon az
új bemenetben és mikor támaszkodjon inkább a korábbi állapotra.

A **KCA-Mamba** (Kalman-Cross-Attention Mamba) ezt explicit Kalman-szűrő
logikával váltja ki: minden időlépésben a hálózat kiszámít egy
$Q$ folyamatzaj- és $R$ mérési zajtermet, ezekből klasszikus Kalman-erősítést
($K$) vezet le, majd az állapotátmeneti faktort ($A = 1 - K$) közvetlenül
ebből állítja elő. Zajos bemenetkor $R \gg Q \Rightarrow K \approx 0 \Rightarrow A \approx 1$
(hosszú memória, kicsi frissítés). Tiszta jelnél $Q \approx R \Rightarrow K \approx 0.5$
(gyors követés).

### 10.2 KCAMambaBlock (egyetlen réteg felépítése)

```
x [B, T, d_model]
  │
  ├─ in_proj ──→ x_cw [B,T,E],  gate = SiLU(x_gate) [B,T,E]
  │                E = H × D  (fejek × állapotméret)
  │
  ├─ conv1d (kauzális, csoportos, kernel=4) → x_core [B,T,E]
  │    ⟶ SiLU aktiváció
  │
  ├─ v_norm(proj_v(x_core)) → v_seq [B,T,H,D]   (értékek fejekre bontva)
  │
  ├─── Kalman-gain számítás ────────────────────────────────────────────────
  │   Q = softplus(Q_net(x_core)) · softplus(q_scale)   [B,T,H,1]
  │   R = softplus(R_net(x_core)) · softplus(r_scale)   [B,T,H,1]
  │
  │   K_base  = Q / (Q + R + ε)          ← klasszikus Kalman-erősítés
  │   K_delta = 0.3 · (σ(K_net) − 0.5)  ← tartalom-függő korrekció, nullában 0
  │   K_seq   = clamp(K_base + K_delta, 1e-4, 0.999)
  │
  │   A = clamp(1 − K_seq, 0.01, 0.99)  ← állapot-megtartási faktor
  │   B = K_seq · v_seq                  ← bemenet-beszivárgási faktor
  │──────────────────────────────────────────────────────────────────────────
  │
  ├─ parallel_scan(A, B, μ₀) → μ_all [B,T,E]
  │    O(T log T) idő, O(T log T) memória, GPU-n párhuzamos
  │
  ├─ out_proj(μ_all · gate) → y_p
  │
  └─ out = y_p + sigmoid(res_gate_bias) · (res_proj(x) − y_p)
             ↑ tanulható reziduális kapu, −1.0 inicializálással (~0.27 induló erősítés)
```

**Tanulható paraméterek inicializálása:**

| Paraméter | Induló érték | Hatás |
|---|---|---|
| `q_scale` | −2.0 | `softplus(−2) ≈ 0.13` → alacsony folyamatzaj |
| `r_scale` | +1.5 | `softplus(+1.5) ≈ 1.73` → magas mérési zaj |
| `res_gate_bias` | −1.0 | `σ(−1) ≈ 0.27` → gyenge reziduális kapu |
| `mu_init` | **0** | tanulható kezdő állapot fejenkénti |

A `q_scale` / `r_scale` inicializálással $K_\text{base} \approx 0.07$, vagyis
$A \approx 0.93$: a modell körülbelül 14 lépés memóriával indul, majd tanítás
közben a feladathoz alkalmazkodik.

### 10.3 Parallel scan

A párhuzamos prefix-scan az asszociativitást kihasználva $O(T)$ szekvenciális
lépés helyett $O(\log T)$ "sweepben" számolja ki az összes rejtett állapotot:

```
Iteráció 1 (step=1):
  A_new[t] = A[t] · A[t−1]
  B_new[t] = B[t] + A_eredeti[t] · B[t−1]

Iteráció 2 (step=2):
  A_new[t] = A[t] · A[t−2]
  B_new[t] = B[t] + A_eredeti[t] · B[t−2]
  ...
```

Fontos: a $B$-frissítésnél mindig az *eredeti* (frissítés előtti) $A$-értéket
kell használni, különben minden korábbi hozzájárulás kétszeresen lecsengne.

### 10.4 KCAMambaStack (több réteg + multi-timescale figyelés)

```
x [B, T, feature_dim]
  │
  ├─ input_proj → input_norm (LayerNorm) → h [B,T, d_hidden]
  │
  ├─ KCAMambaBlock₁ → block_norm₁ → KCAMambaBlock₂ → block_norm₂ → …
  │    N réteg, köztük pre-norm LayerNorm (aktiváció-skála drift megakadályozása)
  │
  ├─ final LayerNorm
  │
  └─ Cross-attention (multi-timescale)
       query = h[:, −1:, :]                      ← utolsó (legújabb) token
       key/value = h[:, ::slow_stride, :]        ← ritka mintavétel (pl. stride=12)
       attn_out → residuális hozzáadás az utolsó pozícióhoz
```

A `slow_stride` (alapértelmezett: 12) egy durvább időléptékű kontextust nyújt
az utolsó tokennek: 1 perces gyertyáknál `stride=12` kb. 12 perces ablakot
jelent.

### 10.5 A teljes rendszer felépítése (hexagonális architektúra)

```
┌─────────────────────────────────────────────────────────────────┐
│                       CORE (üzleti logika)                      │
│                                                                 │
│  core/domain   ←  Event, Signal, Order, Fill, RiskResult        │
│  core/ml       ←  KCAMambaStack, MambaPredictor, Kalman-portok  │
│  core/application ← TradingEngine pipeline:                     │
│      MarketData → StateEstimation → Prediction                  │
│                → Signal → Risk → Execution                      │
└────────────────────────┬────────────────────────────────────────┘
                         │  port-interfészek (ABC)
         ┌───────────────┼───────────────────────┐
         ↓               ↓                       ↓
┌────────────────┐ ┌──────────────┐ ┌────────────────────────────┐
│ adapters/      │ │ adapters/    │ │ adapters/offline_training  │
│ infrastructure │ │ web          │ │                            │
│                │ │              │ │ TrainingEngine:            │
│ Binance WS/REST│ │ aiohttp HTTP │ │  • adatgyűjtés             │
│ RSS + sentiment│ │ + WebSocket  │ │  • KCAMamba tanítás        │
│ EventBus       │ │ dashboard    │ │  • validációs kapu         │
│ ExecutionUseCase│ │ REST API    │ │  • zero-downtime deploy    │
└────────────────┘ └──────────────┘ └────────────────────────────┘
```

A `core/` réteg egyetlen infrastrukturális importot sem tartalmaz; az összes
külső függőség az `adapters/` rétegen keresztül, port-interfészeken át
csatlakozik. Így a teljes kereskedési pipeline szintetikus adatokon is
futtatható (lásd `tests/`).

**Verzió**: 1.0 (2026-04-23) · **Licenc**: az ELTE szakdolgozatra vonatkozó szabályzat szerint

