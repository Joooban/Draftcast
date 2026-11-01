from lolwp.config import ROOT

RUNTIME = {"fastapi", "uvicorn", "pydantic", "pandas", "pyarrow", "numpy", "scikit-learn", "joblib",
           "python-dotenv", "requests"}


def _lines(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_runtime_requirements_minimal():
    lines = _lines("requirements.txt")
    assert all("==" in line for line in lines)
    assert {line.split("==")[0].lower() for line in lines} == RUNTIME


def test_dev_requirements_extend_runtime():
    lines = _lines("requirements-dev.txt")
    assert lines[0] == "-r requirements.txt"
    names = {line.split("==")[0].lower() for line in lines[1:]}
    assert {"lightgbm", "statsmodels", "pytest"} <= names and not names & RUNTIME
