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
   nodes `Flux2Scheduler`, `EmptyFlux2LatentImage` and `ReferenceLatent`. On a
   running instance: `curl -s localhost:8188/object_info | grep -c Flux2Scheduler`.
   If it is `0`, update ComfyUI first — nothing else you do will help.
3. Check free disk. The models total roughly **28 GB**; the Klein checkpoint
   alone is 18 GB. If the volume has less than 35 GB free, stop and say so
   rather than filling the disk.
4. `flux-2-klein-9b.safetensors` is **gated**. Export `HF_TOKEN` from an account
   that has accepted the licence at
   <https://huggingface.co/black-forest-labs/FLUX.2-klein-9b>. Without it that
   one download returns a redirect to the auth wall and everything else still
   succeeds — which is exactly the failure that looks like success. The other
   five models need no token.

## Install

```bash
export HF_TOKEN=hf_...              # required for the gated Klein checkpoint
python scripts/install.py --comfy-root /path/to/ComfyUI
```

The script is resumable and idempotent: it skips a model whose size already
matches the manifest, and it re-runs safely after a network failure. It never
deletes anything.

If you would rather do it by hand, `python scripts/install.py --plan` prints
every clone and download as a shell command without executing them.

## Then verify, and believe only the exit code

```bash
# with ComfyUI running:
python scripts/verify.py --url http://127.0.0.1:8188
# or, without starting a server:
python scripts/verify.py --comfy-root /path/to/ComfyUI --offline
```

Exit codes:

| code | meaning | what to do |
|---|---|---|
| `0` | every node type resolves and every model is in place | done |
| `1` | something is genuinely missing | fix it, re-run |
| `2` | the check could not be performed | this is **not** a pass — find out why |

The `2` is deliberate. A verifier that cannot reach ComfyUI, or cannot read the
manifest, must not be indistinguishable from a verifier that found everything.

Prove the checker itself can fail before you trust a green result:

```bash
python scripts/verify.py --self-test    # must print OK and exit 0
```

## Pip dependencies

The bundled node pack imports `mediapipe`, `pymatting`, `insightface`,
`opencv-python`, `transformers` and `spandrel`. Install into **the interpreter
ComfyUI actually runs with**, which is often not the `python3` on `PATH`:

```bash
/path/to/comfy/venv/bin/python -m pip install -r requirements.txt
```

`insightface` builds native code; on a machine with no compiler, install a
prebuilt wheel instead of letting pip try to build it.

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
cd /path/to/ComfyUI
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()")
echo "using port $PORT"
./venv/bin/python main.py --cpu --port "$PORT" --disable-auto-launch &
# wait for it to finish loading, then:
python scripts/verify.py --url "http://127.0.0.1:$PORT"
kill %1
```

`--cpu` is the point: it touches no VRAM, so it cannot disturb a running job. If
the machine is idle, a plain restart is fine and simpler.

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
