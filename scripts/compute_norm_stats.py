"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import sys
from typing import Any
from typing import get_args
from typing import get_origin

import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _normalize_cli_key(key: str) -> list[str]:
    return [part.replace("-", "_") for part in key.split(".")]


def _field_type(instance: Any, field_name: str) -> Any:
    for field in dataclasses.fields(instance):
        if field.name == field_name:
            return field.type
    raise KeyError(f"Unknown field: {field_name}")


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def _convert_value(value: str, annotation: Any) -> Any:
    if value == "None":
        return None

    origin = get_origin(annotation)
    if origin is not None:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _convert_value(value, args[0])

    if annotation is bool:
        return _parse_bool(value)
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return value

    return value


def _replace_nested(instance: Any, path: list[str], raw_value: str) -> Any:
    field_name = path[0]
    annotation = _field_type(instance, field_name)

    if len(path) == 1:
        return dataclasses.replace(instance, **{field_name: _convert_value(raw_value, annotation)})

    child = getattr(instance, field_name)
    if not dataclasses.is_dataclass(child):
        raise ValueError(f"Cannot override nested field on non-dataclass value: {'.'.join(path)}")
    updated_child = _replace_nested(child, path[1:], raw_value)
    return dataclasses.replace(instance, **{field_name: updated_child})


def _extract_args(argv: list[str]) -> tuple[str, int | None, list[tuple[str, str]]]:
    if not argv or argv[0] in {"-h", "--help"}:
        raise ValueError("help")

    config_name = argv[0]
    max_frames = None
    overrides: list[tuple[str, str]] = []

    i = 1
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {arg}")

        key = arg[2:]
        if "=" in key:
            key, value = key.split("=", 1)
        else:
            i += 1
            if i >= len(argv):
                raise ValueError(f"Missing value for argument: --{key}")
            value = argv[i]

        if key == "max-frames":
            max_frames = int(value)
        else:
            overrides.append((key, value))
        i += 1

    return config_name, max_frames, overrides


def _usage() -> str:
    configs = ", ".join(sorted(_config._CONFIGS_DICT))  # noqa: SLF001
    return (
        "Usage:\n"
        "  python scripts/compute_norm_stats.py <config_name> [--max-frames N] [--field value ...]\n\n"
        "Examples:\n"
        "  python scripts/compute_norm_stats.py pi05_hsr_lora "
        "--data.repo-id /abs/path/to/dataset --assets-base-dir /abs/path/to/assets "
        "--data.assets.asset-id hsr_dataset\n"
        "  python scripts/compute_norm_stats.py pi0_hsr_lora "
        "--data.repo-id lerobot_datasets/task8 --max-frames 50000\n\n"
        f"Available configs:\n  {configs}\n"
    )


def cli(argv: list[str] | None = None) -> tuple[_config.TrainConfig, int | None]:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_name, max_frames, overrides = _extract_args(argv)

    config = _config.get_config(config_name)
    for key, value in overrides:
        config = _replace_nested(config, _normalize_cli_key(key), value)

    return config, max_frames


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config: _config.TrainConfig | str, max_frames: int | None = None):
    if isinstance(config, str):
        config = _config.get_config(config)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    try:
        parsed_config, parsed_max_frames = cli()
    except (KeyError, ValueError) as exc:
        if str(exc) == "help":
            print(_usage())
            raise SystemExit(0) from None
        print(f"Error: {exc}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2) from exc

    main(parsed_config, parsed_max_frames)
