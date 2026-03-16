#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
from typing import Any

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.training import checkpoints as checkpoints_lib
from openpi.training import config as train_config
from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI policy as websocket server for HSR client")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", required=True, help="Train config name (e.g. pi05_hsr)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Optional torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    return parser.parse_args()


def resolve_norm_stats(config: train_config.TrainConfig, checkpoint_dir: Path) -> tuple[dict[str, Any], str]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is not None:
        logging.info("Using norm stats resolved by train config.")
        return data_config.norm_stats, "config"

    asset_id = data_config.asset_id
    if asset_id is None:
        logging.info("No asset_id configured; deferring norm stats resolution to policy_config.")
        return None, None

    candidate_roots: list[tuple[Path, str]] = [(checkpoint_dir / "assets", "checkpoint")]
    for env_name in ("POLICY_DATA_HOME", "OPENPI_DATA_HOME"):
        env_value = os.getenv(env_name)
        if env_value:
            candidate_roots.append((Path(env_value), env_name))

    for root, source_name in candidate_roots:
        try:
            norm_stats = checkpoints_lib.load_norm_stats(root, asset_id)
            logging.info("Using norm stats from %s (%s).", source_name, root / asset_id)
            return norm_stats, source_name
        except FileNotFoundError:
            logging.info("Norm stats not found in %s (%s).", source_name, root / asset_id)

    logging.warning(
        "Norm stats were not found in any candidate location. "
        "Continuing without normalization; policy behavior may differ from training."
    )
    return {}, "disabled"


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    config_name = args.config_name
    config = train_config.get_config(config_name)
    norm_stats, norm_stats_source = resolve_norm_stats(config, checkpoint_dir)

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "config_name": config_name,
            "checkpoint_dir": str(checkpoint_dir),
            "norm_stats_source": norm_stats_source,
            "server_host": args.host,
            "server_port": args.port,
        }
    )

    logging.info("Serving policy config=%s checkpoint=%s on %s:%s", config_name, checkpoint_dir, args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
