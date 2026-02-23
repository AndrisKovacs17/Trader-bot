# Tesztek futtatása

## Miért nem futottak korábban?

A tesztek **nem futottak manuálisan a terminálból**, mert:

1. **Rossz helyen voltak**: A tesztek a workspace root-ban voltak (`a:\Saját\Diplomamunka\`), nem a `diplomamunkakod` mappában
2. **ModuleNotFoundError**: A Python nem találta a `core` modult, mert a `diplomamunkakod` mappa nem volt a Python path-ban

## Megoldás

Most minden teszt tartalmazza ezt a kódrészletet az elején:

```python
import sys
from pathlib import Path

# Add parent directory (diplomamunkakod) to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

Ez automatikusan hozzáadja a `diplomamunkakod` mappát a Python path-hoz, így a `core` modul importálható.

## Tesztek futtatása

### Módszer 1: Egyszerű futtatás (AJÁNLOTT)

Kapcsolódj a `diplomamunkakod` mappába, majd futtasd a tesztet:

```powershell
cd "a:\Saját\Diplomamunka\diplomamunkakod"
python tests\test_fee_fix.py
```

Vagy:

```powershell
cd "a:\Saját\Diplomamunka\diplomamunkakod"
python tests\test_equity.py
python tests\test_fill_fix.py
```

### Módszer 2: Mindegyik teszt egyszerre

```powershell
cd "a:\Saját\Diplomamunka\diplomamunkakod"
Get-ChildItem tests\*.py | Where-Object { $_.Name -ne '__init__.py' } | ForEach-Object { python $_.FullName }
```

### Módszer 3: Python module módban (alternatíva)

Ha pytest van telepítve:

```powershell
cd "a:\Saját\Diplomamunka\diplomamunkakod"
pytest tests/
```

## Tesztek listája

- **test_fee_fix.py**: Ellenőrzi, hogy a fee mindig csökkenti az equity-t (BUY és SELL esetén is)
- **test_equity.py**: Ellenőrzi az equity kalkulációt BUY és SELL fill-ek után
- **test_fill_fix.py**: Ellenőrzi a wallet cash kezelését és position tracking-et

## Eredmények

Minden teszt sikeres (✓) ha:
- Az equity helyesen csökken fee-vel
- BUY és SELL műveletek helyesen frissítik az equity-t
- A pozíciók helyesen követik a tranzakciókat
