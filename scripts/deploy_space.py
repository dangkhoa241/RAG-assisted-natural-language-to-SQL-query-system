"""Deploy the app to a Hugging Face Docker Space.

Uploads only what the Dockerfile needs (the same allowlist as .dockerignore, from the files git tracks), with
space/README.md as the Space's README (its header sets `sdk: docker` and `app_port: 7860`). The Space then
builds the image itself. Files this script manages that no longer exist locally are deleted from the Space.

Log in first (`hf auth login`, or set HF_TOKEN to a write token), then from the repo root:
  python scripts/deploy_space.py --space-id your-username/nl2sql-assistant
Optional, instead of setting them on the website:
  --intent-model your-username/nl2sql-intent-model   sets the INTENT_MODEL_PATH variable
  --secrets-from-env                                 copies GROQ_API_KEY (and CEREBRAS_API_KEY, if set) from
                                                     the environment or .env into Space secrets
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
# Keep in sync with .dockerignore.
INCLUDE_FILES = ["Dockerfile", ".dockerignore", "requirements.txt", "data/healthcare_dataset.csv",
                 "data/retail_sales.csv", "data/saas_subscriptions.csv"]
INCLUDE_DIRS = ["backend/", "src/", "config/", "docs/glossary/", "frontend/"]
EXCLUDE = ["src/model_training.ipynb"]
MANAGED = ["Dockerfile", ".dockerignore", "requirements.txt", "README.md", "backend/**", "src/**", "config/**",
           "docs/**", "data/**", "frontend/**"]
SECRETS = ["GROQ_API_KEY", "CEREBRAS_API_KEY"]


def tracked_files() -> list:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT_DIR, check=True, capture_output=True, text=True).stdout
    files = [f for f in out.splitlines() if f]
    keep = [f for f in files if (f in INCLUDE_FILES or any(f.startswith(d) for d in INCLUDE_DIRS))
            and f not in EXCLUDE]
    missing = [f for f in INCLUDE_FILES if f not in keep]
    if missing:
        sys.exit(f"Not tracked by git (commit them first): {', '.join(missing)}")
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space-id", required=True, help="e.g. your-username/nl2sql-assistant")
    ap.add_argument("--intent-model", help="Hub repo id of the intent model (sets INTENT_MODEL_PATH)")
    ap.add_argument("--secrets-from-env", action="store_true")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    if args.intent_model:
        api.add_space_variable(args.space_id, "INTENT_MODEL_PATH", args.intent_model)
        print(f"Set INTENT_MODEL_PATH={args.intent_model}")
    if args.secrets_from_env:
        from dotenv import dotenv_values

        env = {**dotenv_values(ROOT_DIR / ".env"), **os.environ}
        for name in SECRETS:
            if env.get(name):
                api.add_space_secret(args.space_id, name, env[name])
                print(f"Set secret {name}")   # the value is never printed

    files = tracked_files()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in files + ["space/README.md"]:
            target = stage / ("README.md" if f == "space/README.md" else f)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (ROOT_DIR / f).read_bytes()
            if b"\0" not in data:   # text: a Windows checkout has CRLF, which breaks shell lines in a Linux build
                data = data.replace(b"\r\n", b"\n")
            target.write_bytes(data)
        info = api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=stage,
                                 delete_patterns=MANAGED, commit_message="Deploy from GitHub main")
    owner, name = args.space_id.split("/")
    print(f"Uploaded {len(files) + 1} files ({info.oid[:7] if hasattr(info, 'oid') else info}).")
    print(f"Space:  https://huggingface.co/spaces/{args.space_id}  (build logs under 'Logs')")
    print(f"App:    https://{owner.lower()}-{name.lower().replace('_', '-').replace('.', '-')}.hf.space")


if __name__ == "__main__":
    main()
