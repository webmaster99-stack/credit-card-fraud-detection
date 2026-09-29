import importlib

import fraud.params as params_module


def test_params_path_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "params.yaml"
    custom.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv("FRAUD_PARAMS_PATH", str(custom))
    try:
        reloaded = importlib.reload(params_module)
        assert reloaded.PARAMS_PATH == custom
        assert reloaded.load_params() == {"seed": 7}
    finally:
        monkeypatch.delenv("FRAUD_PARAMS_PATH")
        importlib.reload(params_module)


def test_params_path_defaults_to_repo_root():
    assert params_module.PARAMS_PATH == params_module.REPO_ROOT / "params.yaml"
