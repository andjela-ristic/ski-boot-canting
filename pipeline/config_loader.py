from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_file = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    else:
        config_file = PROJECT_ROOT / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_file}")

    return config