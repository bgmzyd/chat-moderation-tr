"""Store / fetch the trained weights on the Hugging Face Hub (R7: no weights in git).

  python -m src.hub upload  --repo <user>/chat-moderation-tr-models   (needs HF write token)
  python -m src.hub download --repo <user>/chat-moderation-tr-models  (public, no token)
"""
import argparse

from huggingface_hub import HfApi, snapshot_download

from src import config as C


def models_ready() -> bool:
    return all((C.MODEL_DIRS[k] / "threshold.json").exists() for k in C.MODEL_KEYS)


def download(repo=C.HF_REPO_ID):
    print(f"downloading model weights from https://huggingface.co/{repo} ...")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(C.MODELS_DIR))


def upload(repo):
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo, folder_path=str(C.MODELS_DIR), repo_type="model",
                      commit_message="trained models")
    print(f"uploaded to https://huggingface.co/{repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["upload", "download"])
    ap.add_argument("--repo", default=C.HF_REPO_ID)
    a = ap.parse_args()
    upload(a.repo) if a.action == "upload" else download(a.repo)
