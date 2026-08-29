# Bone SR

Czysta **grawitacja N-ciał** z kinematyką **szczególnej teorii względności** (`p = γ m v`).  
Otwarta przestrzeń — **bez ścian**, **bez warstwy społecznej**.

Szczegóły: [`docs/IDEA.md`](docs/IDEA.md).

## Instalacja

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
# GPU (opcjonalnie):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Studio

```bash
python -m bone studio
```

[http://127.0.0.1:8765/](http://127.0.0.1:8765/) — presety **Galaktyka / Gromada / Burst SR**, wybór bryły startowej z listy.

## CLI

```bash
python -m bone --preset galaxy --steps 200 --particles 3000 --out out
python -m bone --preset cluster --geometry 2 --steps 150
```

## Testy

```bash
pytest -q
```
