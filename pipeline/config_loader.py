from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_yaml_file(config_file: Path) -> dict:
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {config_file}")

    return config


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_file = PROJECT_ROOT / "config" / "pipeline_config.yaml"
    else:
        config_file = PROJECT_ROOT / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = load_yaml_file(config_file)

    extends_path = config.pop("extends", None)
    if extends_path is not None:
        parent_file = (config_file.parent / extends_path).resolve()

        if not parent_file.exists():
            raise FileNotFoundError(f"Parent config file not found: {parent_file}")

        parent_config = load_yaml_file(parent_file)
        config = deep_merge_dict(parent_config, config)

    return config
