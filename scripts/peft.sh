# Parameter efficient fine-tuning with PEFT
# https://huggingface.co/docs/lerobot/peft_training

# lerobot-train \
#   --policy.path=lerobot/smolvla_base \
#   --policy.repo_id=lt-s/test_smolvla \
#   --dataset.repo_id=airoa-org/airoa-moma \
#   --dataset.root=/srv/shared/ICRA2026/datasets/airoa-moma \
#   --policy.optimizer_lr=1e-3 \
#   --policy.scheduler_decay_lr=1e-4 \
#   --steps=1000 \
#   --batch_size=32 \
#   --peft.method_type=LORA \
#   --peft.r=64 \
#   --wandb.enable=true


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lerobot-train \
    --config_path="${SCRIPT_DIR}/training_config_pi05_airoa_moma.json"
