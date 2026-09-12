#!/usr/bin/env python3
"""Fetch pinned, trainable CAT weights; no simulation or robot connection.

The existing ONNX file is inference-only. Fetch the original Orbax parameter
tree and configuration from the authors, without replacing any existing file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REPOSITORY = "Axian12138/Click-and-Traverse"
REVISION = "46ce4b57ba0639168d51741b661ff62f7ce6f045"
PREFIX = "logs_v1/generalist_v1/"


def fetch(destination=None):
    from huggingface_hub import HfApi, hf_hub_download
    destination = Path(destination or ROOT/"artifacts/cat_release")
    info = HfApi().model_info(REPOSITORY, revision=REVISION, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.startswith(PREFIX)]
    if info.sha != REVISION or not files:
        raise ValueError("Pinned CAT release is unavailable")
    records = []
    for item in files:
        source = Path(hf_hub_download(REPOSITORY, item.rfilename, revision=REVISION))
        content = source.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if len(content) != item.size:
            raise ValueError("Downloaded CAT file has wrong size")
        if item.lfs:
            if sha != item.lfs.sha256:
                raise ValueError("Downloaded CAT LFS checksum mismatch")
        elif hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest() != item.blob_id:
            raise ValueError("Downloaded CAT Git blob checksum mismatch")
        target = destination/item.rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise FileExistsError("Refusing to replace different existing file: "+str(target))
        else:
            with target.open("xb") as stream, source.open("rb") as original:
                shutil.copyfileobj(original, stream)
        records.append(dict(path=item.rfilename, size=item.size, sha256=sha))
    manifest = dict(repository=REPOSITORY, revision=REVISION, files=records,
        kind="released CAT actor/value/normalizer parameters; not original optimizer state",
        motion_enabled=False)
    text = json.dumps(manifest, indent=2)+"\n"
    output = destination/"manifest.json"
    if output.exists() and output.read_text() != text:
        raise FileExistsError(output)
    if not output.exists():
        with output.open("x") as stream:
            stream.write(text)
    print(json.dumps(dict(manifest=str(output), files=len(files), bytes=sum(r["size"] for r in records))))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Download the pinned published parameter tree")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    else:
        print(json.dumps(dict(repository=REPOSITORY, revision=REVISION, prefix=PREFIX, download_started=False)))
