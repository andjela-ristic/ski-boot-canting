requiremnets:
for scripts:
    python -m pip install pillow pillow-heif

start venv:
    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install pillow pillow-heif
    one time thing: python scripts/convert_heic_to_png.py
    