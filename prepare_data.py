# -*- coding: utf-8 -*-
"""Baixa o ZIP público da UCI ou lê um ZIP local e prepara os 5 .arff.
Não executa o dashboard nem abre links públicos.
"""
from pathlib import Path
import argparse
import urllib.request
import zipfile

UCI_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "polish_bankruptcy"
EXPECTED = {f"{i}year.arff" for i in range(1, 6)}

def prepare(zip_path: Path | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = ROOT / "data" / "polish_companies_bankruptcy_data.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if not zip_path.is_file():
            print("Baixando ZIP oficial do UCI (dataset 365)...")
            urllib.request.urlretrieve(UCI_URL, zip_path)
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for name in archive.namelist():
            base = Path(name).name
            if base in EXPECTED and not name.endswith("/"):
                if base in members:
                    raise ValueError(f"Arquivo duplicado no ZIP: {base}")
                members[base] = name
        missing = EXPECTED - members.keys()
        if missing:
            raise ValueError(f"Arquivos ARFF ausentes no ZIP: {sorted(missing)}")
        for filename, member in members.items():
            # Extrai somente os cinco nomes esperados; não aceita caminhos arbitrários do ZIP.
            (DATA_DIR / filename).write_bytes(archive.read(member))
    print(f"Dados preparados em: {DATA_DIR}")
    print(", ".join(sorted(EXPECTED)))
    return DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar dataset UCI Polish Bankruptcy")
    parser.add_argument("--zip", type=Path, default=None, help="ZIP oficial previamente baixado (opcional)")
    args = parser.parse_args()
    prepare(args.zip)
