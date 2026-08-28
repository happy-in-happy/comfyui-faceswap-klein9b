#!/usr/bin/env python3
"""Install everything manifest.json declares into a ComfyUI tree.

Idempotent and resumable: a model whose size already matches is skipped, a pack
directory that already exists is left alone. Nothing is ever deleted.

    python scripts/install.py --comfy-root /path/to/ComfyUI
    python scripts/install.py --plan            # print the commands, run nothing

Exit codes:
    0  everything the manifest declares is now present
    1  at least one item could not be installed (each one is named)
    2  the install could not start (bad manifest, bad root, no network stack)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

OK, FAILED, CANNOT_START = 0, 1, 2
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def log(msg):
    print(msg, flush=True)


def load_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log("CANNOT START: manifest unreadable: %s" % exc)
        sys.exit(CANNOT_START)


def check_root(root):
    root = os.path.abspath(root)
    if not os.path.isfile(os.path.join(root, "main.py")):
        log("CANNOT START: %s has no main.py, so it is not a ComfyUI root." % root)
        log("Point --comfy-root at the directory holding main.py, models/ and custom_nodes/.")
        sys.exit(CANNOT_START)
    return root


def run(cmd, cwd=None, plan=False):
    if plan:
        log("  $ " + " ".join(cmd))
        return True
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except OSError as exc:
        log("  ! cannot execute %s: %s" % (cmd[0], exc))
        return False
    if proc.returncode != 0:
        log("  ! %s exited %d" % (cmd[0], proc.returncode))
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        for line in tail:
            log("    | " + line)
        return False
    return True


def install_packs(manifest, root, plan):
    custom = os.path.join(root, "custom_nodes")
    if not plan:
        os.makedirs(custom, exist_ok=True)
    problems = []
    for pack in manifest["node_packs"]:
        pid = pack["id"]
        dest = os.path.join(custom, pid)
        if pack["source"] == "bundled":
            src = os.path.join(REPO, pack["path"])
            log("[pack] %s (bundled)" % pid)
            if plan:
                log("  $ cp -r %s %s" % (src, dest))
                continue
            if not os.path.isdir(src):
                problems.append("bundled pack missing from this repo: %s" % src)
                continue
            if os.path.isdir(dest):
                log("  already present, refreshing .py files")
            os.makedirs(dest, exist_ok=True)
            for fn in os.listdir(src):
                if fn.endswith(".py"):
                    shutil.copy2(os.path.join(src, fn), os.path.join(dest, fn))
            log("  installed -> %s" % dest)
            continue

        log("[pack] %s" % pid)
        if os.path.isdir(dest) and not plan:
            # Present is not the same as installed: a pack cloned by someone else,
            # or by an earlier interrupted run, may have none of its Python
            # dependencies. Leave its source alone but make sure its requirements
            # are satisfied, or it will fail to import with nothing having said so.
            log("  already present, left as is; checking its requirements")
            reqs = os.path.join(dest, "requirements.txt")
            if os.path.isfile(reqs):
                if not run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs]):
                    problems.append("pip install failed for pre-existing %s" % pid)
            else:
                log("  no requirements.txt")
            continue
        if not run(["git", "clone", "--depth", "50", pack["repo"], dest], plan=plan):
            problems.append("clone failed: %s" % pack["repo"])
            continue
        pin = pack.get("pinned_commit")
        if pin:
            if not run(["git", "fetch", "--depth", "50", "origin", pin], cwd=dest, plan=plan):
                log("  (shallow fetch of the pin failed; falling back to full fetch)")
                run(["git", "fetch", "--unshallow"], cwd=dest, plan=plan)
            if not run(["git", "checkout", pin], cwd=dest, plan=plan):
                problems.append("cannot check out pinned commit %s in %s" % (pin, pid))
        elif pack.get("registry_version"):
            log("  registry version %s expected; cloned default branch" % pack["registry_version"])
        reqs = os.path.join(dest, "requirements.txt")
        if not plan and os.path.isfile(reqs):
            log("  installing its requirements.txt with %s" % os.path.basename(sys.executable))
            if not run([sys.executable, "-m", "pip", "install", "-q", "-r", reqs]):
                problems.append("pip install failed for %s" % pid)
    return problems


def install_inputs(root, plan):
    """Put the two LoadImage files the graph names into ComfyUI/input/.

    Without them the graph is complete and still cannot be queued: LoadImage
    validates its filename against the contents of input/, so on a fresh machine
    it rejects the prompt exactly the way a missing node would. The shipped files
    are labelled placeholders, not photographs of anyone - they exist so the
    graph loads green and the operator can see which node wants which picture.

    An existing file is never overwritten: by the second run these are usually
    the operator's own photos.
    """
    src_dir = os.path.join(REPO, "assets", "sample_inputs")
    dest_dir = os.path.join(root, "input")
    log("[inputs] LoadImage files -> input/")
    if not os.path.isdir(src_dir):
        return ["sample inputs missing from this repo: %s" % src_dir]
    if plan:
        log("  $ mkdir -p %s" % dest_dir)
        for fn in sorted(os.listdir(src_dir)):
            log("  $ cp -n %s %s" % (os.path.join(src_dir, fn), os.path.join(dest_dir, fn)))
        return []
    os.makedirs(dest_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        dest = os.path.join(dest_dir, fn)
        if os.path.isfile(dest):
            log("  %s already there, left alone" % fn)
            continue
        shutil.copy2(os.path.join(src_dir, fn), dest)
        log("  placed %s (labelled placeholder - replace it with your own photo)" % fn)
    return []


def hf_url(repo, path):
    return "https://huggingface.co/%s/resolve/main/%s" % (repo, path)


def sha256_of(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def digest_problem(model, dest):
    """Return a message if the manifest gives a sha256 and the file disagrees.

    Size alone is a weak identity: this manifest exists partly because a
    different LoRA is published under one of these exact filenames. Size caught
    that twin, but it would not catch same-size corruption or a truncated resume.
    """
    want = model.get("sha256")
    if not want or len(want) != 64:
        return None
    got = sha256_of(dest)
    if got != want:
        return ("%s has the right size but the wrong contents: sha256 %s, manifest says %s"
                % (model["filename"], got[:16] + "...", want[:16] + "..."))
    log("  sha256 verified")
    return None


def download(model, root, plan, token, check_digest=True):
    dest_dir = os.path.join(root, model["dest"].replace("/", os.sep))
    dest = os.path.join(dest_dir, model["filename"])
    want = model.get("size_bytes")
    url = hf_url(model["hf_repo"], model["hf_path"])

    if plan:
        log("  $ mkdir -p %s" % dest_dir)
        log("  $ curl -L%s -o %s \\\n      %s"
            % (" -H \"Authorization: Bearer $HF_TOKEN\"" if model.get("gated") else "", dest, url))
        return None

    if os.path.isfile(dest):
        have = os.path.getsize(dest)
        if want and have == want:
            log("  already present with the expected size")
            return digest_problem(model, dest) if check_digest else None
        log("  present but %d bytes, manifest says %d - redownloading" % (have, want))

    os.makedirs(dest_dir, exist_ok=True)

    if model.get("gated") and not token:
        return ("%s is gated and HF_TOKEN is not set. Accept the licence at "
                "https://huggingface.co/%s and export HF_TOKEN, then re-run."
                % (model["filename"], model["hf_repo"]))

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return ("huggingface_hub is not installed for %s; "
                "run: %s -m pip install huggingface_hub"
                % (os.path.basename(sys.executable), sys.executable))

    try:
        got = hf_hub_download(repo_id=model["hf_repo"], filename=model["hf_path"],
                              token=token or None, local_dir=dest_dir)
    except Exception as exc:  # network, auth, quota - all reported, none swallowed
        return "download failed for %s: %r" % (model["filename"], exc)

    # hf_hub_download preserves the repo-internal path; flatten it to the name
    # the workflow's dropdown expects.
    got = os.path.abspath(got)
    if got != os.path.abspath(dest):
        os.replace(got, dest)
        stray = os.path.dirname(got)
        while os.path.abspath(stray) not in (os.path.abspath(dest_dir), os.path.sep):
            try:
                os.rmdir(stray)
            except OSError:
                break
            stray = os.path.dirname(stray)

    have = os.path.getsize(dest)
    if want and have != want:
        return ("%s downloaded but is %d bytes, manifest says %d"
                % (model["filename"], have, want))
    log("  ok, %d bytes" % have)
    return digest_problem(model, dest) if check_digest else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--comfy-root", help="directory containing main.py, models/, custom_nodes/")
    ap.add_argument("--manifest", default=os.path.join(REPO, "manifest.json"))
    ap.add_argument("--plan", action="store_true", help="print what would run, execute nothing")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-packs", action="store_true")
    ap.add_argument("--skip-inputs", action="store_true",
                    help="do not place the placeholder LoadImage files into input/")
    ap.add_argument("--skip-digest", action="store_true",
                    help="do not sha256 files that already match on size (faster, weaker)")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.manifest)
    if args.plan:
        root = os.path.abspath(args.comfy_root or "/path/to/ComfyUI")
        log("PLAN ONLY - nothing below is executed. Root assumed: %s\n" % root)
    else:
        if not args.comfy_root:
            log("CANNOT START: --comfy-root is required (or use --plan).")
            return CANNOT_START
        root = check_root(args.comfy_root)
        log("ComfyUI root: %s\n" % root)

    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.skip_models:
        log("note: HF_TOKEN is not set. Five of the six models download without it;")
        log("      flux-2-klein-9b.safetensors is gated and will be reported as a failure.\n")

    problems = []
    if not args.skip_packs:
        problems += install_packs(manifest, root, args.plan)

    if not args.skip_models:
        log("")
        for model in manifest["models"]:
            log("[model] %s -> %s" % (model["filename"], model["dest"]))
            err = download(model, root, args.plan, token, check_digest=not args.skip_digest)
            if err:
                log("  ! " + err)
                problems.append(err)

    if not args.skip_inputs:
        log("")
        problems += install_inputs(root, args.plan)

    log("")
    auto = manifest.get("auto_downloaded_on_first_run", [])
    log("%d further assets fetch themselves on the first run; see manifest.json." % len(auto))
    log("The one worth warming by hand is hustvl/vitmatte-small-composition-1k -")
    log("no loader dropdown lists it, so nothing above would have caught its absence.")

    if args.plan:
        log("\nPLAN COMPLETE - re-run without --plan to execute.")
        return OK
    if problems:
        log("\nINSTALL INCOMPLETE - %d problem(s):" % len(problems))
        for p in problems:
            log("  - %s" % p)
        log("\nFix these and re-run; the script skips whatever already succeeded.")
        return FAILED
    log("\nInstall finished. Restart ComfyUI (it enumerates models/ at startup),")
    log("then run:  python scripts/verify.py --url http://127.0.0.1:8188")
    return OK


if __name__ == "__main__":
    sys.exit(main())
