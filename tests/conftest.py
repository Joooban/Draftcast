import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from lolwp.config import ROOT
from synthetic import fill_store


@dataclass
class PipelineDirs:
    data_dir: str
    artifacts_dir: str


def _synthetic(tmp_path_factory) -> PipelineDirs:
    from lolwp.features.build import build
    from lolwp.models import train

    root = tmp_path_factory.mktemp("pipeline")
    data, art = root / "data", root / "artifacts"
    art.mkdir()
    store = fill_store(str(data / "lol.db"), str(art / "champions.json"), per_patch=300)
    build(store, str(art / "champions.json"), str(data), str(art))
    train.train(str(data), str(art))
    try:
        from lolwp.models import insights
        insights.make_insights(str(data), str(art))
    except ImportError:
        pass
    return PipelineDirs(str(data), str(art))


@pytest.fixture(scope="session")
def synthetic_dirs(tmp_path_factory) -> PipelineDirs:
    return _synthetic(tmp_path_factory)


@pytest.fixture(scope="session", params=["synthetic", "real"])
def pipeline_dirs(request, tmp_path_factory) -> PipelineDirs:
    """Trained artifacts to check: always a synthetic run, plus the real one once it exists."""
    if request.param == "real":
        real = PipelineDirs(str(ROOT / "data"), str(ROOT / "artifacts"))
        needed = [Path(real.artifacts_dir) / "draft_model.joblib", Path(real.data_dir) / "draft.parquet"]
        if not all(p.exists() for p in needed):
            pytest.skip("real artifacts not built yet")
        return real
    return request.getfixturevalue("synthetic_dirs")
