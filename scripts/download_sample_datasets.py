from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="airoa-org/airoa-moma",
    repo_type="dataset",          # dataset なので必要
    local_dir="/srv/shared/ICRA2026/datasets/airoa-moma",     # 保存先
    local_dir_use_symlinks=False  # 実体ファイルとして保存したい場合
)

print(f"downloaded to: {local_dir}")