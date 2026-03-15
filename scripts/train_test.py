import os
import pathlib
import subprocess
import sys

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

_RUN_TRAIN_CODE = """
import dataclasses
import sys

from openpi.training import config as _config
from scripts import train

config_name, checkpoint_base_dir, exp_name, resume, num_train_steps = sys.argv[1:]
config = dataclasses.replace(
    _config._CONFIGS_DICT[config_name],  # noqa: SLF001
    batch_size=2,
    num_workers=0,
    checkpoint_base_dir=checkpoint_base_dir,
    exp_name=exp_name,
    overwrite=False,
    resume=(resume == "true"),
    num_train_steps=int(num_train_steps),
    log_interval=1,
    wandb_enabled=False,
)
train.main(config)
"""


def _clean_pythonpath(env: dict[str, str]) -> dict[str, str]:
    current_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    pythonpath = env.get("PYTHONPATH")
    if not pythonpath:
        return env

    filtered = []
    for entry in pythonpath.split(os.pathsep):
        parts = pathlib.Path(entry).parts
        if any(part.startswith("python") and part != current_tag for part in parts):
            continue
        filtered.append(entry)

    if filtered:
        env["PYTHONPATH"] = os.pathsep.join(filtered)
    else:
        env.pop("PYTHONPATH", None)
    return env


def _run_train(config_name: str, checkpoint_base_dir: str, exp_name: str, *, resume: bool, num_train_steps: int) -> None:
    env = _clean_pythonpath(dict(os.environ))
    env["JAX_PLATFORMS"] = "cpu"
    env["OPENPI_DISABLE_ASYNC_CHECKPOINTING"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            _RUN_TRAIN_CODE,
            config_name,
            checkpoint_base_dir,
            exp_name,
            "true" if resume else "false",
            str(num_train_steps),
        ],
        check=True,
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        env=env,
    )


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    checkpoint_base_dir = str(tmp_path / "checkpoint")
    exp_name = "test"

    _run_train(config_name, checkpoint_base_dir, exp_name, resume=False, num_train_steps=2)


@pytest.mark.manual
@pytest.mark.parametrize("config_name", ["debug"])
def test_train_resume(tmp_path: pathlib.Path, config_name: str):
    checkpoint_base_dir = str(tmp_path / "checkpoint")
    exp_name = "test"

    _run_train(config_name, checkpoint_base_dir, exp_name, resume=False, num_train_steps=2)
    _run_train(config_name, checkpoint_base_dir, exp_name, resume=True, num_train_steps=4)
