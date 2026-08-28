# Brief for an installing agent

Paste this whole file to a coding agent that has shell access to the machine
running ComfyUI. It is written to be executable without asking the operator
anything, and to fail loudly instead of half-installing.

---

## Your job

Make the workflow at `workflow/face_swap_klein_9b.json` loadable and runnable on
this machine's ComfyUI. You are done when `scripts/verify.py` exits `0`. You are
not done when it prints something reassuring — read the exit code.

## What you are given

- `manifest.json` — the complete requirement list. Six node packs, six model
  files, four things that download themselves on first run. Treat it as the
  contract; do not improvise substitutes.
- `custom_nodes/happyin_faceswap_nodes/` — twelve custom node classes that ship
  in this repo. There is no external source for them; copy the directory.
- `scripts/install.py` — does the whole install from the manifest.
- `scripts/verify.py` — the acceptance check. Three outcomes, not two.

## Before you touch anything

1. Find the ComfyUI root — the directory that contains `main.py`, `models/` and
   `custom_nodes/`. Do not guess it; confirm `main.py` is there.
2. Confirm the ComfyUI build has FLUX.2 support. The workflow needs the core
   nodes `Flux2Scheduler`, `EmptyFlux2LatentImage` and `ReferenceLatent`. Against
   a running instance, on *its* port — find it with `ss -ltnp | grep python`
   rather than assuming 8188:

   ```bash
   curl -s "http://127.0.0.1:$PORT/object_info" | grep -c Flux2Scheduler
   ```

   If it answers `0`, update ComfyUI first — nothing else you do will help. If it
   answers `401` or refuses the connection, you are not talking to the ComfyUI you
   think you are; find the right port before drawing any conclusion.
3. Check free disk. The models total roughly **28 GB**; the Klein checkpoint
   alone is 18 GB. If the volume has less than 35 GB free, stop and say so
   rather than filling the disk.
4. `flux-2-klein-9b.safetensors` is **gated**. Export `HF_TOKEN` from an account
   that has accepted the licence at
   <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b>. Without it that
   one download returns a redirect to the auth wall and everything else still
   succeeds — which is exactly the failure that looks like success. The other
   five models need no token.

## Find ComfyUI's interpreter first — this is the step that fails silently

Everything below must run with **the Python ComfyUI itself uses**, not the
`python3` on `PATH`. They are routinely different: on one machine this was tested
on, `PATH` gave 3.14 while ComfyUI's virtualenv was 3.12.

It matters twice. `install.py` pip-installs each cloned pack's `requirements.txt`
using `sys.executable`, so running it with the wrong interpreter puts a pack's
dependencies somewhere ComfyUI will never look — **and nothing in the output says
so**. The install appears to succeed and the nodes fail to import later.

```bash
COMFY=/path/to/ComfyUI
PY="$COMFY/.venv/bin/python"          # some trees use venv/, not .venv/
[ -x "$PY" ] || PY="$COMFY/venv/bin/python"
[ -x "$PY" ] || { echo "find ComfyUI's interpreter before continuing"; exit 1; }
"$PY" -V
```

## Install

Dependencies first — `install.py` needs `huggingface_hub` to fetch anything, and
that lives in `requirements.txt`. Run this before the installer, not after:

```bash
"$PY" -m pip install -r requirements.txt
```

`insightface` builds native code; on a machine with no compiler, install a
prebuilt wheel rather than letting pip try to build it.

Then:

```bash
export HF_TOKEN=hf_...              # required for the gated Klein checkpoint
"$PY" scripts/install.py --comfy-root "$COMFY"
```

The script is resumable and idempotent: it skips a model whose size already
matches the manifest, verifies its sha256 when one is recorded, and re-runs safely
after a network failure. It never deletes anything. Pass `--skip-digest` if you
would rather not re-hash 28 GB on a repeat run.

If you would rather do it by hand, `"$PY" scripts/install.py --plan` prints every
clone and download as a shell command without executing them.

## Then verify, and believe only the exit code

```bash
"$PY" scripts/verify.py --url "http://127.0.0.1:$PORT"
```

Exit codes:

| code | meaning | what to do |
|---|---|---|
| `0` | every node type resolves and every model is in place | done |
| `1` | something is genuinely missing | see below — re-running the installer alone will not fix it |
| `2` | the check could not be performed, or could not prove registration | **not** a pass — find out why |

**`--offline` can never return `0`, by design.** It checks that files and pack
directories exist on disk, which is not the same as a node having registered — a
pack can be checked out, complete, and still fail to import. Offline therefore
returns `2` even when everything it can see is in order. Use it to find missing
files quickly; never use it to decide you are finished.

**When you get `1`, do not just re-run the installer.** It skips any pack whose
directory already exists, so a second run changes nothing. Exit `1` names the
missing node type or model; go and find out why *that* one did not register —
usually by importing the pack directly and reading the traceback:

```bash
cd "$COMFY/custom_nodes" && "$PY" -c "import <pack_dir_name>"
```

The `2` is deliberate. A verifier that cannot reach ComfyUI, or cannot read the
manifest, must not be indistinguishable from a verifier that found everything.

Prove the checker itself can fail before you trust a green result:

```bash
python scripts/verify.py --self-test    # must print OK and exit 0
```

## What the bundled pack imports

`mediapipe`, `pymatting`, `insightface`, `opencv-python`, `transformers` and
`spandrel` — all in `requirements.txt`, which you installed with `$PY` before the
installer ran.

`install.py` will also pull in whatever the cloned packs list in their own
`requirements.txt`, including for packs that were already present before you
started. On a machine where a pack directory existed but its dependencies did
not, that is the difference between a node registering and a node failing to
import with nothing having said why.

## After installing, ComfyUI has to be restarted — but maybe not the one that matters

ComfyUI enumerates `models/` and loads `custom_nodes/` at startup. Anything you
just installed is invisible to a process that was already running. If `verify.py`
reports something missing that you know you downloaded, restart before debugging.

**If the ComfyUI on this machine is doing real work, do not bounce it.** You do
not need to. Start a second, throwaway instance from the same tree — it reads the
same `custom_nodes/` and `models/`, so it answers `/object_info` with exactly what
a restarted server would, while the busy one keeps its GPU and its queue:

**Do not hardcode a port — ask the kernel for a free one.** A busy machine has
services you did not put there, and a port that merely *looks* spare may answer
with someone else's `401`. This bit a real run: port 8199 was already taken by an
unrelated authenticated service, so every probe came back Unauthorized.

```bash
cd "$COMFY"
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
echo "using port $PORT"
"$PY" main.py --cpu --port "$PORT" --disable-auto-launch &
# wait for it to finish loading, then:
"$PY" scripts/verify.py --url "http://127.0.0.1:$PORT"
kill %1
```

`--cpu` keeps the throwaway instance off the GPU for its own work. It does **not**
guarantee zero VRAM: measured on a real run, a `--cpu` instance still held 718 MiB
because some custom node in the tree creates a CUDA context at import time. On a
143 GB card next to a job holding 49 GB that is nothing; on a nearly-full card it
is not. Check before you launch:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

If the card has under a gigabyte spare, do not start a second instance at all —
wait for the running job, or verify on a machine that is idle. If the machine is
idle to begin with, a plain restart is fine and simpler.

If `verify.py` exits `2` saying it could not reach the server, check *what* is on
that port before assuming the install failed — `curl -s -o /dev/null -w '%{http_code}'
http://127.0.0.1:$PORT/system_stats` answering `401` means you are talking to
something that is not your ComfyUI.

## Four things that download themselves, and the one that bites

On the first run the graph fetches Florence-2-base-ft into `models/LLM/`,
`u2net.onnx` into `~/.u2net/`, two mediapipe assets into `models/mediapipe/`,
and `hustvl/vitmatte-small-composition-1k` into the HuggingFace cache.

The last one is the trap. Eight of the nine `Happyin_Mask_PersonMask` nodes ask
for `matting_method=VITMatte`, and no loader dropdown lists it, so it is absent
from every "which models do I need" list — including the one you would write by
looking at the workflow. If ComfyUI runs in a container whose HF cache is not on
a persistent volume, it vanishes on every rebuild and the first PersonMask run
stalls on a silent download. Warm it explicitly:

```bash
python -c "from transformers import VitMatteForImageMatting as M; M.from_pretrained('hustvl/vitmatte-small-composition-1k')"
```

## Report back

State, in this order: the ComfyUI root and version you found; the exit code of
`verify.py`; anything you installed that the manifest did not list, and why; and
anything you could not install, naming the exact blocker. If you skipped a step,
say which one — a skipped check is not a passed check.
