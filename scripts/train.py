import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class StepProfileMetrics:
    loader_wait_sec: float
    to_device_sec: float
    fetch_total_sec: float
    fetch_time_sec: float
    compute_sec: float
    train_step_time_sec: float
    total_iter_sec: float
    total_iter_time_sec: float
    samples_per_sec: float


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _make_step_profile(
    batch_size: int,
    loader_profile: _data_loader.BatchProfile | None,
    *,
    fetch_time_sec: float,
    train_step_time_sec: float,
) -> StepProfileMetrics:
    loader_wait_sec = loader_profile.loader_wait_sec if loader_profile is not None else fetch_time_sec
    to_device_sec = loader_profile.to_device_sec if loader_profile is not None else 0.0
    fetch_total_sec = loader_profile.fetch_total_sec if loader_profile is not None else fetch_time_sec
    total_iter_time_sec = fetch_time_sec + train_step_time_sec
    samples_per_sec = batch_size / total_iter_time_sec if total_iter_time_sec > 0 else float("inf")

    return StepProfileMetrics(
        loader_wait_sec=loader_wait_sec,
        to_device_sec=to_device_sec,
        fetch_total_sec=fetch_total_sec,
        fetch_time_sec=fetch_time_sec,
        compute_sec=train_step_time_sec,
        train_step_time_sec=train_step_time_sec,
        total_iter_sec=total_iter_time_sec,
        total_iter_time_sec=total_iter_time_sec,
        samples_per_sec=samples_per_sec,
    )


def _format_step_profile(metrics: StepProfileMetrics) -> str:
    return ", ".join(f"{field.name}={getattr(metrics, field.name):.6f}" for field in dataclasses.fields(metrics))


def _format_profile_summary(metrics: list[StepProfileMetrics]) -> str:
    summary = []
    for field in dataclasses.fields(StepProfileMetrics):
        values = np.asarray([getattr(metric, field.name) for metric in metrics], dtype=np.float64)
        summary.append(f"{field.name}_avg={values.mean():.6f}")
        summary.append(f"{field.name}_min={values.min():.6f}")
        summary.append(f"{field.name}_max={values.max():.6f}")
    return ", ".join(summary)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    num_samples = data_loader.num_samples()
    steps_per_epoch = data_loader.steps_per_epoch()
    if num_samples is not None and steps_per_epoch is not None:
        logging.info(f"Dataset size: {num_samples:,} samples, {steps_per_epoch:,} steps/epoch")
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    profile_metrics: list[StepProfileMetrics] = []
    profile_summary_logged = False
    profile_measure_limit = config.profile_warmup_steps + config.profile_measure_steps
    if config.profile_performance:
        logging.info(
            "Performance profiling enabled: warmup_steps=%d, measure_steps=%d",
            config.profile_warmup_steps,
            config.profile_measure_steps,
        )
        data_loader.reset_profile()

    infos = []
    for step in pbar:
        if config.profile_performance:
            train_step_start = time.perf_counter()
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            # JAX dispatch is asynchronous, so synchronize here to measure the actual device compute time instead of
            # the enqueue latency seen by Python.
            train_state, info = jax.block_until_ready((train_state, info))
            train_step_time_sec = time.perf_counter() - train_step_start
        else:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if steps_per_epoch:
                reduced_info["epoch"] = (step + 1) / steps_per_epoch
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if config.profile_performance:
            fetch_start = time.perf_counter()
            batch = next(data_iter)
            fetch_time_sec = time.perf_counter() - fetch_start
            step_profile = _make_step_profile(
                config.batch_size,
                data_loader.latest_profile(),
                fetch_time_sec=fetch_time_sec,
                train_step_time_sec=train_step_time_sec,
            )
            relative_step = step - start_step
            if relative_step < config.profile_warmup_steps:
                if relative_step == 0 and config.profile_warmup_steps > 0:
                    logging.info(
                        "Performance profiling warmup started: first %d step(s) are excluded from the summary.",
                        config.profile_warmup_steps,
                    )
            elif relative_step < profile_measure_limit:
                profile_metrics.append(step_profile)
                logging.info(
                    "Performance profile step %d/%d (train step=%d): %s",
                    len(profile_metrics),
                    config.profile_measure_steps,
                    step,
                    _format_step_profile(step_profile),
                )
                if len(profile_metrics) == config.profile_measure_steps and not profile_summary_logged:
                    logging.info(
                        "Performance profile summary: measured_steps=%d, %s",
                        len(profile_metrics),
                        _format_profile_summary(profile_metrics),
                    )
                    profile_summary_logged = True
            elif relative_step == profile_measure_limit and config.profile_measure_steps == 0 and not profile_summary_logged:
                logging.info("Performance profile summary: measured_steps=0")
                profile_summary_logged = True
        else:
            batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    if config.profile_performance and not profile_summary_logged:
        if profile_metrics:
            logging.info(
                "Performance profile summary (partial): measured_steps=%d/%d, %s",
                len(profile_metrics),
                config.profile_measure_steps,
                _format_profile_summary(profile_metrics),
            )
        else:
            logging.info("Performance profile summary: no measured steps were collected.")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
