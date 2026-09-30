"""Carregamento dos dados públicos UCI Polish Companies Bankruptcy.

Preserva o mapeamento de horizontes e as conversões da V3.
Execute `python prepare_data.py` na raiz do projeto antes de carregar os dados.
A base UCI é experimental e falência não equivale automaticamente a default contratual.
"""

from pathlib import Path

import pandas as pd
from scipy.io import arff


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "polish_bankruptcy"

# Na nomenclatura empregada na V3, 5year.arff é associado ao
# horizonte experimental aproximado de 12 meses e 1year.arff ao de 60 meses.
HORIZON_FILES = {
    12: "5year.arff",
    24: "4year.arff",
    36: "3year.arff",
    48: "2year.arff",
    60: "1year.arff",
}


def load_arff(horizon_months=12, data_dir=None):
    """Carrega os indicadores financeiros e o alvo de um horizonte.

    Parameters
    ----------
    horizon_months : int
        Um dos horizontes experimentais 12, 24, 36, 48 ou 60 meses.
    data_dir : str | pathlib.Path | None
        Diretório com os cinco ARFF. O padrão coincide com prepare_data.py.

    Returns
    -------
    tuple[pandas.DataFrame, str]
        Dados carregados e nome do ARFF selecionado, como na função da V3.
    """
    if horizon_months not in HORIZON_FILES:
        raise ValueError(
            f"Horizonte inválido: {horizon_months}. "
            f"Utilize um de: {tuple(HORIZON_FILES)}."
        )

    folder = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    filename = HORIZON_FILES[horizon_months]
    filepath = folder / filename
    if not filepath.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {filepath}. "
            "Execute 'python prepare_data.py' na raiz do repositório "
            "ou informe data_dir com os ARFF originais."
        )

    data, _ = arff.loadarff(str(filepath))
    df = pd.DataFrame(data)

    for column in [c for c in df.columns if c.startswith("Attr")]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "class" not in df.columns:
        raise ValueError(f"Coluna 'class' não encontrada em {filename}.")

    df["class"] = df["class"].apply(
        lambda value: int(value.decode())
        if isinstance(value, (bytes, bytearray))
        else int(value)
    )
    return df, filename
