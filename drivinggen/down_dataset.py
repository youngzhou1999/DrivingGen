from huggingface_hub import login, snapshot_download
import os

login(token="your_token", add_to_git_credential=True)

snapshot_download(
    repo_id="yangzhou99/DrivingGen",
    repo_type="dataset",          # ✅ 关键：用 repo_type
    local_dir="./data",
    local_dir_use_symlinks=False, # ✅ 更稳（尤其是集群/共享盘）
)