
from pathlib import Path

import pandas as pd

from src import utils


def test_basic_info_smoke(capsys):
    """basic_info should print without raising."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    utils.basic_info(df)
    captured = capsys.readouterr()
    # Should contain row / column information
    assert "Rows:" in captured.out


def test_numeric_summary():
    """numeric_summary returns expected statistics dataframe."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    summary = utils.numeric_summary(df, ["A"])
    assert summary.loc["A", "mean"] == 2


def test_load_data(tmp_path):
    """load_data reads a sample pipe-delimited file correctly."""
    sample = "col1|col2\n1|2020-01-01\n2|2020-02-01\n"
    tmp_file = tmp_path / "sample.txt"
    tmp_file.write_text(sample)
    df = utils.load_data(tmp_file, datetime_cols=["col2"], delimiter="|")
    assert df.shape == (2, 2)
    assert pd.api.types.is_datetime64_any_dtype(df["col2"])
