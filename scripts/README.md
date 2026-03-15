# ファインチューニング実行方法

このドキュメントは、[`scripts/train.py`](/home/keito-sonehara/workspace/ICRA2026/airoa-evaluation-ICRA/scripts/train.py) を使って HSR データで `pi0` / `pi05` を LoRA ファインチューニングする手順をまとめたものです。

## 前提

- Python 3.11 系を使います。
- 依存関係は `uv` で入れる想定です。
- 学習には `jax` が必要です。GPU 学習を行う場合は CUDA 対応の JAX 実行環境を用意してください。
- データセット指定は `--data.repo-id` で行います。この値は内部で `LeRobotDataset(...)` に渡されます。
  - Hugging Face の `repo_id`
  - LeRobot が解決できるローカルデータセット識別子やパス
  のどちらでも扱う想定です。

セットアップ例:

```bash
uv sync --extra dev
```

## 最初にやること: 正規化統計の計算

この実装では、通常の学習前に `state` と `actions` の正規化統計が必要です。  
初回の HSR データセットに対しては、先に [`scripts/compute_norm_stats.py`](/home/keito-sonehara/workspace/ICRA2026/airoa-evaluation-ICRA/scripts/compute_norm_stats.py) を実行してください。

`compute_norm_stats.py` も `train.py` と同様に override できます。少なくとも次は学習時と揃えて指定するのが安全です。

- `--data.repo-id`
- `--assets-base-dir`
- 必要に応じて `--data.assets.assets-dir`
- 必要に応じて `--data.assets.asset-id`
- 必要に応じて `--data.action-mode`
- 必要に応じて `--data.convert-gripper`

### `pi0` 用の統計量を計算

```bash
export DATASET_PATH=/abs/path/to/hsr_lerobot_dataset
export ASSETS_BASE=/abs/path/to/assets
export ASSET_ID=hsr_lerobot_dataset  # 統計量を保存するときの名前。repo-id とは別に短い名前を付ける。

uv run python scripts/compute_norm_stats.py pi0_hsr_lora \
  --data.repo-id "${DATASET_PATH}" \
  --assets-base-dir "${ASSETS_BASE}" \
  --data.assets.asset-id "${ASSET_ID}"
```

### `pi05` 用の統計量を計算

```bash
export DATASET_PATH=/abs/path/to/hsr_lerobot_dataset
export ASSETS_BASE=/abs/path/to/assets
export ASSET_ID=hsr_lerobot_dataset  # 統計量を保存するときの名前。repo-id とは別に短い名前を付ける。

uv run python scripts/compute_norm_stats.py pi05_hsr_lora \
  --data.repo-id "${DATASET_PATH}" \
  --assets-base-dir "${ASSETS_BASE}" \
  --data.assets.asset-id "${ASSET_ID}"
```

出力先は次になります。

```text
<assets-base-dir>/<config名>/<data.assets.asset-id>/norm_stats.json
```

学習時は、ここで使った `--assets-base-dir` と `--data.assets.asset-id` を同じ値で渡すと、この統計量が自動で読まれます。

## 基本構文

```bash
uv run python scripts/train.py <config名> \
  --exp-name <実験名> \
  --data.repo-id <データセット指定> \
  --checkpoint-base-dir <出力先の親ディレクトリ>
```

実際のチェックポイント出力先は次のように決まります。

```text
<checkpoint-base-dir>/<config名>/<exp-name>/
```

例えば `--checkpoint-base-dir /work/checkpoints --exp-name run1` で `pi05_hsr_lora` を実行すると、出力先は次になります。

```text
/work/checkpoints/pi05_hsr_lora/run1/
```

## HSR での LoRA 実行例

以下では、データセットと出力先を環境変数で分けて指定しています。

### `pi0` を HSR で LoRA ファインチューニング

```bash
export DATASET_PATH=/abs/path/to/hsr_lerobot_dataset
export OUTPUT_BASE=/abs/path/to/checkpoints
export ASSETS_BASE=/abs/path/to/assets
export ASSET_ID=hsr_lerobot_dataset  # compute_norm_stats.py 実行時と同じ値にする。

uv run python scripts/train.py pi0_hsr_lora \
  --exp-name pi0_hsr_lora_run1 \
  --data.repo-id "${DATASET_PATH}" \
  --assets-base-dir "${ASSETS_BASE}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --data.assets.asset-id "${ASSET_ID}"
```

### `pi05` を HSR で LoRA ファインチューニング

```bash
export DATASET_PATH=/abs/path/to/hsr_lerobot_dataset
export OUTPUT_BASE=/abs/path/to/checkpoints
export ASSETS_BASE=/abs/path/to/assets
export ASSET_ID=hsr_lerobot_dataset  # compute_norm_stats.py 実行時と同じ値にする。

uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_run1 \
  --data.repo-id "${DATASET_PATH}" \
  --assets-base-dir "${ASSETS_BASE}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --data.assets.asset-id "${ASSET_ID}"
```

## よく使うハイパーパラメータ

`train.py` は Tyro ベースの CLI なので、設定値はコマンドラインからそのまま上書きできます。

### 学習全体

- `--batch-size`
  - グローバルバッチサイズです。
- `--num-train-steps`
  - 学習ステップ数です。
- `--num-workers`
  - DataLoader の worker 数です。
- `--seed`
  - 乱数 seed です。
- `--log-interval`
  - 何ステップごとにログを出すかです。
- `--save-interval`
  - 何ステップごとに checkpoint を保存するかです。
- `--keep-period`
  - 一定周期の checkpoint を残します。
- `--fsdp-devices`
  - FSDP の shard 数です。`jax.device_count()` を割り切る必要があります。
- `--resume True`
  - 既存の checkpoint から再開します。
- `--overwrite True`
  - 既存の出力先を上書きします。`resume` と同時指定はできません。

例:

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_bs128 \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --batch-size 128 \
  --num-train-steps 100000 \
  --num-workers 16 \
  --log-interval 20 \
  --save-interval 2000 \
  --fsdp-devices 2
```

### 学習率スケジュール

デフォルトは `CosineDecaySchedule` です。以下のようにネストした項目を上書きできます。

- `--lr-schedule.warmup-steps`
- `--lr-schedule.peak-lr`
- `--lr-schedule.decay-steps`
- `--lr-schedule.decay-lr`

例:

```bash
uv run python scripts/train.py pi0_hsr_lora \
  --exp-name pi0_hsr_lora_lr \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --lr-schedule.warmup-steps 1000 \
  --lr-schedule.peak-lr 5e-5 \
  --lr-schedule.decay-steps 200000 \
  --lr-schedule.decay-lr 5e-6
```

### オプティマイザ

デフォルトは `AdamW` です。主に次を上書きできます。

- `--optimizer.b1`
- `--optimizer.b2`
- `--optimizer.eps`
- `--optimizer.weight-decay`
- `--optimizer.clip-gradient-norm`

例:

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_opt \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --optimizer.weight-decay 1e-6 \
  --optimizer.clip-gradient-norm 0.5
```

### HSR データ変換まわり

HSR 用 config では、データ変換の挙動も CLI から変更できます。

- `--data.action-mode`
  - `relative`
  - `absolute_arm_head_relative_gripper_base`
  - `state_diff_arm_head_relative_gripper_base`
- `--data.adapt-to-pi`
  - HSR 空間を pi 系の内部空間へ合わせるかどうかです。
- `--data.convert-gripper`
  - グリッパ表現の変換を入れるかどうかです。
- `--data.base-action-dim`
  - base action の次元数です。

例:

```bash
uv run python scripts/train.py pi0_hsr_lora \
  --exp-name pi0_hsr_lora_state_diff \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --data.action-mode state_diff_arm_head_relative_gripper_base \
  --data.convert-gripper True
```

### アセットと正規化統計

正規化統計などのアセットを明示したい場合は、次を使えます。

- `--assets-base-dir`
  - config ごとのアセット基準ディレクトリです。
- `--data.assets.assets-dir`
  - アセットを直接置いたディレクトリです。
- `--data.assets.asset-id`
  - アセット識別子です。

例:

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_assets \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --data.assets.assets-dir /abs/path/to/assets/pi05_hsr_lora \
  --data.assets.asset-id my_hsr_dataset
```

## WandB の使い方

この学習スクリプトは WandB に対応しています。`wandb_enabled` のデフォルト値は `True` です。

### 有効化して使う

事前にログインします。

```bash
wandb login
```

その上で通常どおり実行すれば、`project_name` と `exp_name` が WandB に反映されます。

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_wandb \
  --project-name openpi-hsr \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}"
```

補足:

- Run 名は `exp_name` になります。
- Project 名は `--project-name` で変えられます。
- 再開時は出力先ディレクトリ内の `wandb_id.txt` を使って同じ run に resume します。

### 無効化する

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_nowandb \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --wandb-enabled False
```

## よくある調整パターン

### バッチサイズを下げて安全に始める

```bash
uv run python scripts/train.py pi0_hsr_lora \
  --exp-name pi0_hsr_lora_safe \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --batch-size 32 \
  --num-workers 4 \
  --fsdp-devices 1
```

### 中断した学習を再開する

同じ `config名`、`exp_name`、`checkpoint-base-dir` を使って再実行します。

```bash
uv run python scripts/train.py pi05_hsr_lora \
  --exp-name pi05_hsr_lora_run1 \
  --data.repo-id "${DATASET_PATH}" \
  --checkpoint-base-dir "${OUTPUT_BASE}" \
  --resume True
```

## 補足

- `pi0_hsr_lora` と `pi05_hsr_lora` は、どちらも LoRA 用の freeze 設定を使う HSR 向け config です。
- `pytorch_weight_path` は現在の [`scripts/train.py`](/home/keito-sonehara/workspace/ICRA2026/airoa-evaluation-ICRA/scripts/train.py) では使われていません。
- ベースモデルの重み初期化は config 内の `weight_loader` で行われます。
