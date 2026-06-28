from omegaconf import OmegaConf

from worldcup.utils.paths import project_root


def test_test_set_override_in_config():
    cfg = OmegaConf.load(project_root() / "configs" / "config.yaml")
    assert "test_set" in cfg
    assert cfg.test_set is None
