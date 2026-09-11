#!/usr/bin/env python3
"""Fetch a small pinned official terrain baseline; never run downloaded models.

Only the terrain checkpoint/config and one paired robot/object/USD example per
requested family are fetched. Existing files must match upstream hashes. Model
loading belongs in a separate, simulation-only validation step.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PROVENANCE = json.loads((ROOT / "provenance.json").read_text())
REPO = PROVENANCE["dataset_repository"]
REV = PROVENANCE["dataset_revision"]
API = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"


def safe_path(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in path:
        raise ValueError(f"Unsafe artifact path: {path!r}")
    return p


def portable_asset_reference(ref, stem):
    """Only relocate known upstream absolute texture paths, never arbitrary files."""
    if ref.startswith("/mnt/") and "/object_usd/textures/" in ref:
        suffix = ref.split("/object_usd/textures/", 1)[1]
        return str(safe_path("textures/" + stem + "/" + suffix))
    return str(safe_path(ref))


def entries(folder):
    result = []
    url = API + quote(str(safe_path(folder)), safe="/") + "?limit=1000"
    while url:
        with urlopen(url, timeout=60) as response:
            result.extend(json.load(response))
            match = re.search(r'<([^>]+)>; rel="next"', response.headers.get("Link", ""))
            url = match.group(1) if match else None
            if url and not url.startswith(API):
                raise ValueError("Unexpected pagination origin")
    return result


def record(entry):
    return {"path": str(safe_path(entry["path"])), "size": entry["size"],
            "hash_kind": "sha256" if entry.get("lfs") else "git-blob-sha1",
            "hash": entry["lfs"]["oid"] if entry.get("lfs") else entry["oid"]}


def digest(path, kind):
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git-blob-sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(path, item):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlink artifact: {item['path']}")
    if path.stat().st_size != item["size"] or digest(path, item["hash_kind"]) != item["hash"]:
        raise ValueError(f"Artifact checksum mismatch: {item['path']}")


def fetch(item, destination):
    path = destination / safe_path(item["path"])
    if path.exists() or path.is_symlink():
        verify_file(path, item)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    # Only our incomplete download is replaced; completed artifacts are immutable.
    if partial.is_symlink():
        raise ValueError("Refusing symlink partial download")
    print(f"Fetching {item['path']} ({item['size']/1e6:.1f} MB)", flush=True)
    with urlopen(BASE + quote(item["path"], safe="/"), timeout=60) as response, partial.open("wb") as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            out.write(chunk)
    verify_file(partial, item)
    partial.rename(path)
    return path


def build_manifest(families, destination):
    items = [record(x) for x in entries("checkpoint/SONIC/models/terrain_release") if x["type"] == "file"]
    checkpoint = next(x for x in items if x["path"] == PROVENANCE["checkpoint_path"])
    if checkpoint["hash"] != PROVENANCE["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from provenance pin")
    scenes = []
    for family in families:
        if family not in {"stair_p1", "stair_p2", "curb", "slope", "sitting"}:
            raise ValueError("Unsupported terrain family")
        base = f"data/{family}"
        robots = sorted((x for x in entries(base + "/robot") if x["path"].endswith(".pkl")
                         and not x["path"].endswith("metadata.pkl")), key=lambda x: x["path"])
        robot = robots[0]
        stem = PurePosixPath(robot["path"]).stem
        items.append(record(robot))
        usd = None
        for folder, extension in (("objects", ".pkl"), ("object_usd", ".usd")):
            matches = [x for x in entries(f"{base}/{folder}") if PurePosixPath(x["path"]).name == stem + extension]
            if len(matches) != 1:
                raise ValueError(f"Missing paired {folder} for {stem}")
            item = record(matches[0])
            items.append(item)
            if folder == "object_usd":
                usd = fetch(item, destination)
        # USD may be a binary crate. Use OpenUSD rather than parsing as text.
        from pxr import UsdUtils
        layers, assets, payloads = UsdUtils.ExtractExternalReferences(str(usd))
        refs = [*layers, *assets, *payloads]
        item["asset_remaps"] = {}
        item["missing_optional_textures"] = []
        for ref in sorted(set(refs)):
            if ref == "OmniPBR.mdl":
                continue  # Built-in Isaac Sim material, not a dataset file.
            relative = safe_path(portable_asset_reference(ref, stem))
            if str(relative) != ref:
                item["asset_remaps"][ref] = str(relative)
            target = PurePosixPath(base) / "object_usd" / relative
            matches = [x for x in entries(str(target.parent)) if x["path"] == str(target)]
            if not matches and ref.startswith("/mnt/") and target.name in {"model_metallic.jpg", "model_roughness.jpg"}:
                # The public dataset omits these appearance-only maps. Explicitly
                # clear them in derived scenes; never change collision geometry.
                item["asset_remaps"][ref] = ""
                item["missing_optional_textures"].append(ref)
                continue
            if len(matches) != 1:
                raise ValueError(f"Unresolved USD dependency: {target}")
            items.append(record(matches[0]))
        scenes.append({"family": family, "stem": stem})
    items = list({x["path"]: x for x in items}.values())
    return {"repository": REPO, "revision": REV, "scenes": scenes, "files": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "fetch", "verify"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "data_manifest.json")
    parser.add_argument("--destination", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--families", nargs="+", default=["stair_p1", "curb", "slope", "sitting"])
    args = parser.parse_args()
    if args.action == "prepare":
        if args.manifest.exists():
            raise SystemExit("Manifest exists: use fetch or verify")
        manifest = build_manifest(args.families, args.destination)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = json.loads(args.manifest.read_text())
    if manifest["repository"] != REPO or manifest["revision"] != REV:
        raise ValueError("Manifest provenance mismatch")
    for item in manifest["files"]:
        if args.action == "verify":
            verify_file(args.destination / safe_path(item["path"]), item)
        else:
            fetch(item, args.destination)
    print(f"Verified {len(manifest['files'])} artifacts; simulation NOT started", flush=True)


if __name__ == "__main__":
    main()
