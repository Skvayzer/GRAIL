#!/usr/bin/env python3
"""Create a lightweight, private context ZIP; never start a simulator or policy.

Only documented source/evidence locations are read. Credentials, environments,
model weights, arrays, meshes and Git databases are never packaged. Large JSON
evidence is explicitly summarized; original source files remain unchanged.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
USER_ROOT = ROOT.parents[1]
BASE = "aa31d8242ac79b11545b9e3635f73014a227bdfc"
REPOS = {
    "GRAIL-CAT": ROOT,
    "Click-and-Traverse": ROOT.parent / "Click-and-Traverse",
    "humanoid_navigation": ROOT.parent / "humanoid_navigation",
}
BINARY_SUFFIXES = {
    ".pt", ".pth", ".onnx", ".npz", ".npy", ".pkl", ".pickle", ".bin",
    ".so", ".a", ".o", ".dll", ".exe", ".lib", ".pyc", ".stl", ".obj",
    ".dae", ".glb", ".gltf", ".usd", ".usdc", ".png", ".jpg", ".jpeg",
    ".gif", ".mp4", ".webm", ".mov", ".pdf", ".zip", ".gz", ".7z",
    ".woff", ".woff2", ".ttf", ".ico", ".blend", ".mcap", ".db3",
}
EVIDENCE_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".log"}
PRIVATE_NAMES = {".netrc", ".env", "hosts.yml", "credentials", "id_rsa", "id_ed25519"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".cache", "tmp", "wandb"}
SECRET_PATTERNS = [
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:api[_-]?key|access[_-]?token|password)\s*[=:]\s*[\"']?[a-zA-Z0-9+/]{40,}"),
]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(value, location="$", omitted=None):
    """Bound huge evaluation episode arrays; retain ordinary metadata verbatim."""
    if omitted is None:
        omitted = []
    if isinstance(value, dict):
        return {k: compact(v, location + "." + k, omitted) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 128:
            omitted.append({"json_path": location, "original_count": len(value)})
            return {"_handoff_omitted_array": True, "count": len(value),
                    "first_two_items": [compact(v, location + "[]", omitted) for v in value[:2]]}
        return [compact(v, location + "[]", omitted) for v in value]
    return value


class Package:
    def __init__(self, output):
        self.output = output
        self.prefix = output.stem
        self.partial = output.with_name(output.name + ".partial")
        self.z = zipfile.ZipFile(self.partial, "x", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True)
        self.members = []
        self.names = set()
        self.exclusions = []
        self.categories = collections.Counter()
        self.updated = time.monotonic()

    def exclude(self, path, reason):
        p = Path(path)
        self.exclusions.append(dict(source=str(p), reason=reason,
                                    bytes=p.stat().st_size if p.is_file() and not p.is_symlink() else None))

    def add(self, name, data, category, source=None, mode=0o644):
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or name in self.names:
            raise ValueError("Unsafe or duplicate member: " + name)
        if any(part in PRIVATE_NAMES for part in p.parts) or p.suffix in {".key", ".pem"}:
            raise ValueError("Credential-like path blocked: " + name)
        if not name.endswith((".png", ".pdf")) and any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError("Credential-like content blocked; inspect privately: " + name)
        info = zipfile.ZipInfo(self.prefix + "/" + name, time.localtime()[:6])
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | mode) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        self.z.writestr(info, data, compresslevel=6)
        self.names.add(name)
        self.members.append(dict(path=name, bytes=len(data), sha256=sha(data),
                                 category=category, source=str(source) if source else "generated"))
        self.categories[category] += len(data)
        if time.monotonic() - self.updated > 15:
            print(f"Packaged {len(self.members)} files; {self.partial.stat().st_size / 2**20:.1f} MiB ZIP", flush=True)
            self.updated = time.monotonic()

    def json(self, name, value, category):
        self.add(name, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode(), category)

    def text_file(self, path, name, category, limit=2**20, summarize=False):
        if path.is_symlink():
            self.exclude(path, "symlink omitted; restore through recorded repository"); return
        if path.name in PRIVATE_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
            self.exclude(path, "excluded binary or credential-like file"); return
        n = path.stat().st_size
        if n > limit and not (summarize and path.suffix == ".json" and n < 200*2**20):
            self.exclude(path, f"text-size cap {limit} bytes"); return
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            self.exclude(path, "non-UTF8/binary file"); return
        if b"\x00" in data:
            self.exclude(path, "binary NUL bytes"); return
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            # Third-party headers can embed example private-key syntax. Omit
            # and disclose the entire file rather than weaken the credential
            # check or mutate source to make the archive look complete.
            self.exclude(path, "credential-like content or embedded key example; not exported"); return
        if n > limit:
            removed = []
            reduced = compact(json.loads(data), omitted=removed)
            summary = dict(derived_summary=True, original_source=str(path),
                           original_bytes=n, original_sha256=sha(data),
                           omitted_arrays=removed, data=reduced)
            self.json(name + ".summary.json", summary, category)
            self.exclude(path, "large JSON replaced by labeled array-trimmed summary")
        else:
            self.add(name, data, category, path, stat.S_IMODE(path.stat().st_mode))


def metadata(path):
    def text(*args):
        return git(path, *args).decode().strip()
    urls = {}
    for name in text("remote").splitlines():
        url = text("remote", "get-url", name)
        if re.search(r"https?://[^/]*@", url):
            raise ValueError("Credential-bearing Git remote; refusing to export")
        urls[name] = url
    return dict(path=str(path), head=text("rev-parse", "HEAD"),
                branch=text("branch", "--show-current"), remotes=urls,
                status=text("status", "--short"),
                refs=text("for-each-ref", "--format=%(refname) %(objectname)"))


def add_source(pkg, name, path):
    record = metadata(path)
    # A context snapshot must not silently lose uncommitted work.
    if record["status"]:
        raise ValueError(f"{name} is dirty; preserve/review changes before packaging")
    pkg.json(f"git/{name}/repository.json", record, "git-context")
    pkg.add(f"git/{name}/commits.txt", git(path, "log", "--all", "--date=iso-strict",
            "--format=%H %ad %s", "-n", "150"), "git-context")
    modes = []
    for entry in git(path, "ls-files", "-s", "-z").split(b"\x00"):
        if not entry:
            continue
        info, relative = entry.decode().split("\t", 1)
        mode = info.split()[0]
        target = path / relative
        if mode in {"120000", "160000"}:
            modes.append(dict(path=relative, git_mode=mode, git_object=info.split()[1],
                              link=os.readlink(target) if target.is_symlink() else None))
            continue
        if target.is_file():
            pkg.text_file(target, f"source/{name}/{relative}", "source", limit=5*2**20)
    pkg.json(f"git/{name}/symlinks-and-submodules.json", modes, "git-context")
    return record


def walk_evidence(pkg, base, prefix):
    for directory, dirs, files in os.walk(base, followlinks=False):
        for excluded in [d for d in dirs if d in SKIP_DIRS or (Path(directory)/d).is_symlink()]:
            pkg.exclude(Path(directory)/excluded, "cache, W&B internal directory, or symlink omitted")
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (Path(directory)/d).is_symlink()]
        for filename in sorted(files):
            path = Path(directory)/filename
            if path.suffix.lower() in EVIDENCE_SUFFIXES:
                pkg.text_file(path, prefix + "/" + path.relative_to(base).as_posix(),
                              "saved-evidence", summarize=True)
            else:
                pkg.exclude(path, "heavy model/array/geometry/media or non-evidence extension")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.suffix != ".zip" or output.exists() or output.with_name(output.name + ".partial").exists():
        parser.error("Choose a new .zip path; existing archive/partial files are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    pkg = Package(output)
    try:
        for p in sorted((ROOT/"research/handoff").glob("*.md")):
            pkg.text_file(p, p.name, "handoff-guide")
        records = {name: add_source(pkg, name, path) for name, path in REPOS.items()}
        pkg.add("git/GRAIL-CAT/implementation.patch", git(ROOT, "diff", "--binary", BASE, "HEAD"), "git-context")
        pkg.add("git/GRAIL-CAT/research-series.mbox", git(ROOT, "format-patch", "--stdout", "--binary",
                BASE + "..HEAD"), "git-context")
        for p in sorted((USER_ROOT/"research/research").glob("*.md")):
            if p.name.startswith(("GRAIL-CAT -", "Click-and-Traverse -", "Obstacle-Aware Loco-Manipulation -")):
                pkg.text_file(p, "notes/" + p.name, "proposal-notes")
        walk_evidence(pkg, ROOT/"research/runs", "evidence/runs")
        walk_evidence(pkg, ROOT/"research/artifacts", "evidence/artifacts")
        report = USER_ROOT/"Desktop/GRAIL_CAT_Report_20260914"
        for name in ["GRAIL_CAT_Implementation_Report_20260914.pdf", "report_data.json", "manifest.json", "report_text.md"]:
            p = report/name
            pkg.add("reports/" + name, p.read_bytes(), "report", p)
        figures = [
            "GRAIL_CAT_review_20260911_v2/01_isaac_frozen_stairs_cat.png",
            "GRAIL_CAT_review_20260911_v2/02_guidance_before_after.png",
            "GRAIL_CAT_environment_review_20260912_v2/02_arm_clearance_closeup_conflict.png",
            "CAT_training_layouts_20260912/lateral_310018.png",
            "CAT_training_layouts_20260912/low_320018.png",
            "CAT_training_layouts_20260912/overhead_330018.png",
            "CAT_training_layouts_20260912/mixed_340018.png",
        ]
        for name in figures:
            p = USER_ROOT/"Desktop"/name
            pkg.add("figures/" + name, p.read_bytes(), "figure", p)
        final = ROOT/"research/runs/20260914_cat_generated_full_25344_noeval_v1"
        provenance = json.loads((ROOT/"research/provenance.json").read_text())
        latest = json.loads((final/"latest.json").read_text())
        pkg.json("ASSET_RECOVERY.json", dict(
            runtime_included=False,
            grail_pretrained=provenance,
            grail_pretrained_local=str(ROOT/"research/artifacts"/provenance["checkpoint_path"]),
            final_checkpoint=dict(local=str(final/latest["checkpoint"]), sha256=latest["sha256"],
                                  bytes=(final/latest["checkpoint"]).stat().st_size,
                                  externally_uploaded=False, role="failed legacy run, not adapter-v2 initialization"),
            generated_bank=dict(local=str(ROOT/"research/runs/20260912_cat_generated_bank_v2"),
                                manifest_in_zip="evidence/runs/20260912_cat_generated_bank_v2/bank.json",
                                arrays_included=False, generator="source/GRAIL-CAT/research/cat_parallel_bank.py"),
            other_runtime_locations=[str(ROOT/"research/artifacts"), str(ROOT/"research/runs"),
                                     str(ROOT/"imports/SONIC/gear_sonic/data/assets"),
                                     str(REPOS["Click-and-Traverse"]/"data")],
            isaaclab=dict(repository="https://github.com/isaac-sim/IsaacLab", revision=provenance["isaaclab_revision"]),
            credentials="Not included. Re-authenticate independently; do not paste tokens into project files.",
            relocation="Historical configs contain original absolute paths; use explicit new configs, preserve evidence."),
            "recovery")
        pkg.json("EXCLUSIONS.json", dict(
            global_exclusions=["virtual environments", "Git object databases", "credentials and home configs",
                               "model checkpoints", "large arrays and meshes", "videos", "unrelated projects"],
            entries=pkg.exclusions), "inventory")
        manifest = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        lightweight=True, simulator_started=False, evaluation_run=False, robot_contacted=False,
                        source_repositories=records, members=pkg.members.copy(),
                        raw_bytes_by_category=dict(pkg.categories),
                        security="Allowlisted locations; common credential names/token signatures blocked; no secret-store export.")
        pkg.json("MANIFEST.json", manifest, "manifest")
    finally:
        pkg.z.close()
    if pkg.partial.stat().st_size > 40*2**20:
        raise RuntimeError("ZIP exceeds lightweight 40 MiB cap; reduce selection before finalizing")
    # Verify all member data, not a model/simulator evaluation.
    with zipfile.ZipFile(pkg.partial) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate ZIP names")
        for member in manifest["members"]:
            data = archive.read(pkg.prefix + "/" + member["path"])
            if sha(data) != member["sha256"]:
                raise RuntimeError("Member hash mismatch: " + member["path"])
    pkg.partial.rename(output)
    archive_hash = sha(output.read_bytes())
    output.with_name(output.name + ".sha256").write_text(archive_hash + "  " + output.name + "\n")
    print(json.dumps(dict(zip=str(output), bytes=output.stat().st_size,
                         mib=round(output.stat().st_size/2**20, 2), files=len(pkg.members),
                         sha256=archive_hash, verified=True), indent=2), flush=True)


if __name__ == "__main__":
    main()
