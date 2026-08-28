# Face Swap — FLUX.2 Klein 9B (ComfyUI)

Everything needed to load and run this workflow on a machine that has never
seen it: the twelve custom nodes it depends on, pinned versions of the five
public node packs, exact download sources for all six model files, and a
checker that tells you whether the install actually worked.

The graph: 190 nodes, 279 links, 53 distinct node types, nothing muted or
bypassed. It swaps a head and face from a reference image onto a target,
keeping the target's lighting and skin tone, then routes the result through
person-masking, colour transfer, paste-back and a skin-detail upscale.

## Quick start

```bash
git clone https://github.com/happy-in-happy/comfyui-faceswap-klein9b
cd comfyui-faceswap-klein9b

export HF_TOKEN=hf_...                       # only the Klein checkpoint is gated
python scripts/install.py --comfy-root /path/to/ComfyUI

# restart ComfyUI, then:
python scripts/verify.py --url http://127.0.0.1:8188
```

`verify.py` exiting `0` is the definition of done. See
[AGENT_BRIEF.md](AGENT_BRIEF.md) if you want to hand this to a coding agent
rather than run it yourself — it is written to be pasted verbatim.

## Actually running it

Installing is not running. Once `verify.py` exits `0`:

1. **Open ComfyUI** and load `workflow/face_swap_klein_9b.json` — drag the file
   onto the canvas, or use Workflow → Open.
2. **Put your two photos in `ComfyUI/input/`.** `install.py` has already placed
   two labelled placeholders there so the graph loads without red nodes; they are
   cards, not faces, and a run with them produces nothing useful. Replace them, or
   point the nodes at your own files:

   | node | title | which photo |
   |---|---|---|
   | 103 | `images_original` | the **target**: the person whose body, lighting and skin tone stay |
   | 105 | `images_reference` | the **reference**: the face and head that get swapped on |

   The filenames the graph ships with are `result_00.jpg` (node 103) and
   `input_00.jpg` (node 105). The names are historical and read backwards — go by
   the node titles, not the filenames.
3. **Queue it.** The first run is slow and mostly silent: Florence-2, `u2net`,
   the mediapipe assets and the VITMatte weights all download on demand, roughly
   750 MB in total, before any pixel is produced.

If nothing appears to start at all, look at the ComfyUI console rather than the
canvas. A graph that fails validation is refused before it reaches the queue, and
the browser shows very little — the console prints the exact node and value:

```
Failed to validate prompt for output 723:
* Happyin_Mask_PersonMask 212:
  - Value not in list: crop_mode: '%' not in ['crop', 'disabled']
```

That specific failure is fixed in this repo (see below), but the shape recurs
whenever a graph meets a different version of a node. `verify.py` now predicts it.

## What gets installed

### Node packs

| Pack | Provides | Source |
|---|---|---|
| `happyin_faceswap_nodes` | 12 classes, 37 instances | **bundled in this repo** |
| `ComfyUI-Florence2` | `DownloadAndLoadFlorence2Model`, `Florence2Run` | [kijai](https://github.com/kijai/ComfyUI-Florence2), registry 1.0.8 |
| `ComfyUI_LayerStyle` | `LayerUtility: ImageScaleByAspectRatio V2` | [chflame163](https://github.com/chflame163/ComfyUI_LayerStyle), registry 2.0.38 |
| `ComfyUI_essentials` | `MaskPreview+` | [cubiq](https://github.com/cubiq/ComfyUI_essentials), registry 1.1.0 |
| `was-node-suite-comfyui` | `Image Blank`, `Image Rembg`, `Text Concatenate`, `Text Multiline`, `Upscale Model Loader` | [WASasquatch](https://github.com/WASasquatch/was-node-suite-comfyui) @ `ea935d10` |
| `Comfyui-PainterFluxImageEdit` | `PainterFluxImageEdit` | [princepainter](https://github.com/princepainter/Comfyui-PainterFluxImageEdit) @ `34f94bd5` |

The remaining 31 node types are ComfyUI core — but they include `Flux2Scheduler`,
`EmptyFlux2LatentImage` and `ReferenceLatent`, so **a build with FLUX.2 support
is mandatory**. The graph was authored on ComfyUI 0.13.0 (recorded in
`extra.node_versions` inside the JSON); this requirement list was verified
against a running ComfyUI 0.20.1 / Python 3.12 / torch 2.11.

`extra.node_versions` also names three packs the graph never instantiates —
`ComfyUI-Easy-Use`, `ComfyUI-GGUF`, `comfyui-inpaint-nodes`. That is leftover
metadata from the authoring machine's install, not a dependency.
`manifest.json` is the contract: install what it lists, nothing more.

### Models — about 28 GB

| File | Goes to | Source |
|---|---|---|
| `flux-2-klein-9b.safetensors` (18.2 GB) | `models/diffusion_models/` | [black-forest-labs/FLUX.2-klein-9b](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b) — **gated** |
| `qwen_3_8b_fp8mixed.safetensors` (8.7 GB) | `models/text_encoders/` | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b) |
| `flux2-vae.safetensors` (336 MB) | `models/vae/` | same repo, `split_files/vae/` |
| `klein/Realism_Engine_Klein_V2.safetensors` (1.09 GB) | `models/loras/klein/` | [Rarriemf/realismengineklein](https://huggingface.co/Rarriemf/realismengineklein) |
| `klein/f2k_consist_20260225.safetensors` (331 MB) | `models/loras/klein/` | [happyinhappy/f2k-consist-klein9b-lora](https://huggingface.co/happyinhappy/f2k-consist-klein9b-lora) — **custom, trained for this workflow** |
| `1x-ITF-SkinDiffDetail-Lite-v1.pth` (20 MB) | `models/upscale_models/` | [uwg/upscaler](https://huggingface.co/uwg/upscaler), `ESRGAN/` |

Klein uses a **Qwen3-8B** text encoder loaded through `CLIPLoader` with type
`flux2` — not the Mistral encoder FLUX.2-dev takes. Getting this wrong is the
most common way to make the graph load and then produce noise.

Note on the custom LoRA: a **different** 570 MB file exists on HuggingFace under
the identical filename `f2k_consist_20260225.safetensors`. It is not this one and
will not reproduce the result. The manifest pins size and origin so the installer
cannot pick up the wrong twin.

### Four things nothing tells you about

These download themselves on first run and appear in no loader dropdown:
Florence-2-base-ft into `models/LLM/`, `u2net.onnx` into `~/.u2net/`, two
mediapipe assets into `models/mediapipe/`, and
`hustvl/vitmatte-small-composition-1k` into the HuggingFace cache.

The last one is the interesting one. Eight of the nine `Happyin_Mask_PersonMask`
instances run `matting_method=VITMatte`, which reaches it through a bare
`from_pretrained` — so it lands in the HF cache rather than under `models/`. On a
containerised ComfyUI whose HF cache is not on a persistent volume it disappears
on every rebuild, and two workers behind one dispatcher can silently disagree
about whether the workflow runs. Warm it explicitly:

```bash
python -c "from transformers import VitMatteForImageMatting as M; M.from_pretrained('hustvl/vitmatte-small-composition-1k')"
```

## The checker

`scripts/verify.py` derives its requirement list **from the workflow JSON on
every run** instead of keeping a second copy in the script. A hand-maintained
gate drifts away from the graph it guards; this one cannot.

It predicts what ComfyUI's own validator would say, so a `0` means the graph can
actually be queued — not merely that the pieces are on disk. Three things it
checks that a "do the files exist" check cannot:

- **Widget arity.** If a node has gained an input since the graph was saved, the
  saved array is one short and every value after the new input is read one slot
  early. ComfyUI refuses the whole prompt; nothing reaches the queue. This
  happened to this very workflow — see the fix below.
- **Enum values.** Every dropdown value is checked against what the installed
  node actually accepts, accounting for the invisible `control_after_generate`
  slot the frontend stores after a seed.
- **Operator inputs.** The two `LoadImage` filenames are reported separately,
  because they depend on your photos rather than on the install. `--require-inputs`
  makes them blocking when you want run-readiness rather than install-readiness.

It reports three outcomes rather than two:

| exit | meaning |
|---|---|
| `0` | every node type resolves, every model is visible to its loader |
| `1` | something is genuinely missing, each item named |
| `2` | the check could not run — **not** a pass |

The separate `2` matters: a checker that cannot reach ComfyUI must not look
identical to one that found everything. `--self-test` runs five controls, three
of which must come back red, so a green result is evidence the instrument works
rather than evidence it is asleep.

```bash
python scripts/verify.py --self-test
```

Online it asks a running ComfyUI's `/object_info`, which is what the workflow's
dropdowns actually resolve against — stronger than listing directories.
`--offline --comfy-root ...` falls back to files on disk and says plainly that it
has not proven node registration, only file presence.

## Repository layout

```
manifest.json                        machine-readable contract: packs, models, sizes, hashes
AGENT_BRIEF.md                       paste-to-an-agent install instructions
workflow/face_swap_klein_9b.json     the graph
custom_nodes/happyin_faceswap_nodes/ the 12 bundled node classes + helpers
scripts/install.py                   manifest -> installed tree; --plan for a dry run
scripts/verify.py                    acceptance check, three-state
requirements.txt                     pip deps of the bundled pack
```

## Licence

MIT for the tooling and the bundled node pack (see [LICENSE](LICENSE)). The
third-party packs keep their own licences; model weights keep theirs, including
the FLUX.2 Klein licence you accept on HuggingFace before downloading.
