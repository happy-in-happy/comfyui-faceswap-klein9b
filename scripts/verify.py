#!/usr/bin/env python3
"""Acceptance check for the Face Swap (FLUX.2 Klein 9B) workflow.

Three outcomes, never two:

    0  PASS     every node type resolves and every model file is in place
    1  FAIL     something is genuinely missing
    2  UNKNOWN  the check could not be performed

The third code exists because a checker that cannot reach ComfyUI, or cannot
read the workflow, must not be indistinguishable from one that found
everything. Absence of a failure signal is not a pass.

The requirement list is derived from the workflow JSON on every run rather than
written down here, so it cannot drift away from the graph it is supposed to
guard. Ask the workflow what it needs; do not maintain a second copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PASS, FAIL, UNKNOWN = 0, 1, 2

# Node types that exist only in the browser UI and are never registered
# server-side. Absent from /object_info by design, not by breakage.
FRONTEND_ONLY = {"Reroute", "Note", "MarkdownNote", "PrimitiveNode"}

# loader class -> (widget index holding the filename, /object_info input name,
#                  directory under models/)
LOADERS = {
    "UNETLoader": (0, "unet_name", "diffusion_models"),
    "CLIPLoader": (0, "clip_name", "text_encoders"),
    "VAELoader": (0, "vae_name", "vae"),
    "LoraLoaderModelOnly": (0, "lora_name", "loras"),
    "Upscale Model Loader": (0, "model_name", "upscale_models"),
    "CheckpointLoaderSimple": (0, "ckpt_name", "checkpoints"),
    "ControlNetLoader": (0, "control_net_name", "controlnet"),
}

# Some model roots have a historical alias; either satisfies the requirement.
DIR_ALIASES = {
    "diffusion_models": ["diffusion_models", "unet"],
    "text_encoders": ["text_encoders", "clip"],
}


class Unknown(Exception):
    """The check could not be performed. Distinct from a failed check."""


def read_workflow(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            wf = json.load(fh)
    except OSError as exc:
        raise Unknown("cannot read workflow %s: %s" % (path, exc))
    except ValueError as exc:
        raise Unknown("workflow %s is not valid JSON: %s" % (path, exc))
    nodes = wf.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise Unknown("workflow %s carries no nodes array" % path)
    return nodes


def requirements(nodes):
    """Return (node_types, models) demanded by the graph itself."""
    types, models = set(), {}
    for n in nodes:
        t = n.get("type")
        if not t:
            continue
        types.add(t)
        spec = LOADERS.get(t)
        if not spec:
            continue
        idx, field, subdir = spec
        widgets = n.get("widgets_values") or []
        if len(widgets) > idx and isinstance(widgets[idx], str) and widgets[idx]:
            models[(t, field, subdir, widgets[idx])] = None
    return sorted(types - FRONTEND_ONLY), sorted(models)


def fetch_object_info(url, timeout):
    endpoint = url.rstrip("/") + "/object_info"
    try:
        with urllib.request.urlopen(endpoint, timeout=timeout) as resp:
            if resp.status != 200:
                raise Unknown("%s returned HTTP %s" % (endpoint, resp.status))
            return json.load(resp)
    except urllib.error.URLError as exc:
        raise Unknown("cannot reach %s: %s" % (endpoint, exc.reason))
    except (OSError, ValueError) as exc:
        raise Unknown("cannot read %s: %r" % (endpoint, exc))


def enum_for(object_info, cls, field):
    try:
        value = object_info[cls]["input"]["required"][field][0]
    except (KeyError, IndexError, TypeError):
        return None
    return value if isinstance(value, list) else None


# The datatypes that render as a widget. Everything else arrives over a link and
# occupies no widgets_values slot.
#
# This is deliberately a list of what IS a widget, not a list of what is a link.
# The other way round looks equivalent and is not: every custom node pack invents
# its own link types (FL2MODEL, CROP_DATA, ...), and any type missing from a
# denylist would be silently counted as a widget, shifting every value after it.
# That mistake cost a false accusation against a perfectly good Florence2Run node.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def widget_slots(object_info, cls):
    """The widgets_values slots a node's required inputs occupy, in order.

    Each entry is (input_name, enum_or_None). A seed-like input declares
    `control_after_generate` in its metadata, and the frontend then stores a
    SECOND value right after it ('randomize', 'fixed', ...). That extra slot is
    real: ignore it and every value after a seed reads one position early, which
    is indistinguishable from genuine drift.
    """
    try:
        required = object_info[cls]["input"]["required"]
    except (KeyError, TypeError):
        return None
    slots = []
    for name, value in required.items():
        if not isinstance(value, list) or not value:
            continue
        spec = value[0]
        meta = value[1] if len(value) > 1 and isinstance(value[1], dict) else {}
        if isinstance(spec, list):
            slots.append((name, spec))
        elif isinstance(spec, str) and spec in WIDGET_TYPES:
            slots.append((name, None))
        else:
            continue
        if meta.get("control_after_generate"):
            slots.append((name + ".control_after_generate", None))
    return slots


def widget_arity(object_info, cls):
    """How many widgets_values entries a node's required inputs demand."""
    slots = widget_slots(object_info, cls)
    return None if slots is None else len(slots)


def check_enum_values(object_info, nodes, operator_inputs=("LoadImage",)):
    """Predict the enum rejections ComfyUI would raise, before you queue anything.

    Returns (defects, operator). `operator` holds values that depend on files the
    person running this supplies -- LoadImage names -- which are missing on a
    fresh install by definition and are not a fault in the repository.
    """
    defects, operator = [], []
    for n in nodes:
        cls = n.get("type")
        vals = n.get("widgets_values")
        if cls not in object_info or not isinstance(vals, list):
            continue
        slots = widget_slots(object_info, cls)
        if not slots or len(vals) < len(slots):
            continue  # arity is reported separately; mapping would be meaningless
        for (name, enum), value in zip(slots, vals):
            if enum is None or value in enum:
                continue
            line = ("node %s (%s) %s = %r is not one of the %d values the installed "
                    "node accepts" % (n.get("id"), cls, name, value, len(enum)))
            (operator if cls in operator_inputs else defects).append(line)
    return defects, operator


def check_widget_arity(object_info, nodes):
    """Catch saved widget arrays that are SHORTER than the node now requires.

    Measured 2026-08-28: a node gained a `crop_mode` input, so every value saved
    after it shifted one slot and ComfyUI rejected the whole prompt with
    prompt_outputs_failed_validation - nothing ever reached the queue. Every node
    type resolved and every model was present, so a check that only looks at
    those reports a clean install for a graph that cannot run.

    Only "too short" is a fault. Arrays are routinely LONGER than the required
    count: the frontend appends control_after_generate to seed and int widgets,
    and optional inputs take slots too. Flagging those would drown the real one.
    """
    problems = []
    for n in nodes:
        cls = n.get("type")
        vals = n.get("widgets_values")
        if cls not in object_info or not isinstance(vals, list):
            continue
        want = widget_arity(object_info, cls)
        if want is None or len(vals) >= want:
            continue
        problems.append(
            "node %s (%s) saved %d widget values but the installed node requires %d - "
            "the graph was saved against a different version of that node and ComfyUI "
            "will refuse the prompt" % (n.get("id"), cls, len(vals), want)
        )
    return problems


def check_online(object_info, node_types, models, nodes=None):
    problems = []
    missing_nodes = [t for t in node_types if t not in object_info]
    for t in missing_nodes:
        problems.append("node type not registered: %s" % t)
    notes = []
    if nodes:
        problems.extend(check_widget_arity(object_info, nodes))
        defects, operator = check_enum_values(object_info, nodes)
        problems.extend(defects)
        notes.extend(operator)
    for cls, field, _subdir, name in models:
        if cls in missing_nodes:
            continue  # already reported; its enum cannot exist
        enum = enum_for(object_info, cls, field)
        if enum is None:
            problems.append("loader %s has no %s enum to check" % (cls, field))
        elif name not in enum:
            problems.append("model not visible to %s.%s: %s" % (cls, field, name))
    return problems, len(node_types), len(models), notes


def check_offline(comfy_root, node_types, models):
    root = os.path.abspath(comfy_root)
    if not os.path.isfile(os.path.join(root, "main.py")):
        raise Unknown("%s does not look like a ComfyUI root (no main.py)" % root)
    models_dir = os.path.join(root, "models")
    if not os.path.isdir(models_dir):
        raise Unknown("%s has no models/ directory" % models_dir)

    problems = []
    for _cls, _field, subdir, name in models:
        candidates = DIR_ALIASES.get(subdir, [subdir])
        if not any(os.path.isfile(os.path.join(models_dir, c, name)) for c in candidates):
            problems.append(
                "model file absent: models/{%s}/%s" % ("|".join(candidates), name)
            )

    # Offline we cannot ask ComfyUI whether a class registered, only whether the
    # pack directory is present. A present directory is not a loaded node: a pack
    # can be checked out, intact, and still fail to import. This mode therefore
    # never returns PASS - see check_offline's caller.
    custom = os.path.join(root, "custom_nodes")
    if not os.path.isdir(custom):
        problems.append("custom_nodes/ directory absent")
    else:
        present = {d.lower() for d in os.listdir(custom)}
        for want, needle in (
            ("happyin_faceswap_nodes", "happyin"),
            ("ComfyUI-Florence2", "florence2"),
            ("ComfyUI_LayerStyle", "layerstyle"),
            ("ComfyUI_essentials", "essential"),
            ("was-node-suite-comfyui", "was-node-suite"),
            ("Comfyui-PainterFluxImageEdit", "painterflux"),
        ):
            if not any(needle in d for d in present):
                problems.append("node pack directory absent: custom_nodes/%s" % want)
    return problems, 0, len(models)


def report(problems, n_nodes, n_models, mode, can_pass=True, notes=(), notes_block=False):
    print("mode: %s" % mode)
    if n_nodes:
        print("node types required by the workflow: %d" % n_nodes)
    print("model files required by the workflow: %d" % n_models)
    if notes:
        print()
        print("input images the operator must supply (%d) - not a fault in this repo:" % len(notes))
        for note in notes:
            print("  - %s" % note)
        print("  Put your own photos in ComfyUI/input/ and point the two LoadImage")
        print("  nodes at them, or run install.py which copies labelled placeholders.")
        if notes_block:
            problems = list(problems) + list(notes)
    if problems:
        print()
        print("FAIL - %d problem(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return FAIL
    print()
    if not can_pass:
        # Measured, 2026-08-28: a tree where every file and directory was in place
        # still failed online, because a third-party pack's module was committed
        # without a .py extension and could not import. Offline saw a directory
        # whose name matched and would have said PASS. A mode that cannot observe
        # registration must not be allowed to bless it.
        print("UNKNOWN - every file and pack directory is in place, but this mode")
        print("cannot observe whether the node classes actually registered. A pack")
        print("can be checked out, intact, and still fail to import.")
        print("This is not a pass. Re-run against a running ComfyUI with --url.")
        return UNKNOWN
    print("PASS - every requirement the workflow declares is satisfied.")
    return PASS


def self_test():
    """Negative controls. A checker that cannot go red proves nothing."""
    failures = []

    nodes = [
        {"type": "UNETLoader", "widgets_values": ["m.safetensors", "default"]},
        {"type": "Happyin_Mask_PersonMask", "widgets_values": []},
        {"type": "Reroute"},
    ]
    types, models = requirements(nodes)
    if types != ["Happyin_Mask_PersonMask", "UNETLoader"]:
        failures.append("frontend-only node not excluded, or types wrong: %r" % (types,))
    if len(models) != 1:
        failures.append("model extraction wrong: %r" % (models,))

    good = {
        "UNETLoader": {"input": {"required": {"unet_name": [["m.safetensors"]]}}},
        "Happyin_Mask_PersonMask": {"input": {"required": {}}},
    }
    probs, _, _, _ = check_online(good, types, models)
    if probs:
        failures.append("control 1 (everything present) should be clean, got %r" % probs)

    # must go red: node class absent
    probs, _, _, _ = check_online({k: v for k, v in good.items() if k != "Happyin_Mask_PersonMask"}, types, models)
    if not probs:
        failures.append("control 2 (missing node) stayed green")

    # must go red: model absent from the enum
    bad = json.loads(json.dumps(good))
    bad["UNETLoader"]["input"]["required"]["unet_name"] = [["other.safetensors"]]
    probs, _, _, _ = check_online(bad, types, models)
    if not probs:
        failures.append("control 3 (missing model) stayed green")

    # must go red: loader present but carrying no enum at all
    bad2 = json.loads(json.dumps(good))
    bad2["UNETLoader"]["input"]["required"]["unet_name"] = ["STRING"]
    probs, _, _, _ = check_online(bad2, types, models)
    if not probs:
        failures.append("control 4 (enum absent) stayed green")

    # must be UNKNOWN, not FAIL and not PASS: unreadable workflow
    try:
        read_workflow(os.path.join(os.path.dirname(__file__), "does-not-exist.json"))
        failures.append("control 5 (absent workflow) did not raise Unknown")
    except Unknown:
        pass

    # control 6a: a saved widget array shorter than the node now requires must
    # be caught. This is what a node gaining an input does to an old graph.
    oi_arity = {"N": {"input": {"required": {
        "img": ["IMAGE"], "a": ["BOOLEAN"], "b": [["x", "y"]], "c": ["INT"]}}}}
    short = [{"id": 1, "type": "N", "widgets_values": [True, "x"]}]          # 2 < 3
    exact = [{"id": 2, "type": "N", "widgets_values": [True, "x", 5]}]       # 3 == 3
    longer = [{"id": 3, "type": "N", "widgets_values": [True, "x", 5, "randomize"]}]
    if not check_widget_arity(oi_arity, short):
        failures.append("control 6a: a too-short widget array was not caught")
    if check_widget_arity(oi_arity, exact):
        failures.append("control 6a: an exact-length widget array was flagged")
    if check_widget_arity(oi_arity, longer):
        failures.append("control 6a: control_after_generate made a valid array look wrong")

    # control 6: the offline mode must never be able to award a PASS, because it
    # cannot see whether a node registered. Regression guard for a measured
    # incident on 2026-08-28, where offline reported PASS on a tree whose online
    # check reported FAIL at the same moment.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        offline_clean = report([], 0, 6, "offline (test)", can_pass=False)
        online_clean = report([], 2, 6, "online (test)", can_pass=True)
    text = buf.getvalue()
    if offline_clean != UNKNOWN:
        failures.append("control 6: clean offline check returned %r, must be UNKNOWN(2)" % offline_clean)
    if "PASS" in text.split("UNKNOWN")[0]:
        failures.append("control 6: offline printed PASS before its UNKNOWN")
    if online_clean != PASS:
        failures.append("control 6: clean online check returned %r, must be PASS(0)" % online_clean)

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  - %s" % f)
        return FAIL
    print("SELF-TEST OK - 7 controls: 3 mandatory-red, the widget-arity guard, the offline-cannot-PASS guard")
    return PASS


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_wf = os.path.join(os.path.dirname(here), "workflow", "face_swap_klein_9b.json")

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8188",
                    help="running ComfyUI to interrogate (default: %(default)s)")
    ap.add_argument("--comfy-root", help="ComfyUI directory, for --offline")
    ap.add_argument("--offline", action="store_true",
                    help="check files on disk instead of asking a running ComfyUI")
    ap.add_argument("--workflow", default=default_wf)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--require-inputs", action="store_true",
                    help="also fail when the LoadImage inputs are missing (run-readiness, not install-readiness)")
    ap.add_argument("--self-test", action="store_true",
                    help="prove this checker can report failure, then exit")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    can_pass = True
    try:
        nodes = read_workflow(args.workflow)
        node_types, models = requirements(nodes)
        if args.offline:
            if not args.comfy_root:
                raise Unknown("--offline needs --comfy-root")
            problems, n_nodes, n_models = check_offline(args.comfy_root, node_types, models)
            notes = []
            mode = "offline (files on disk; node registration NOT proven, so PASS is not available)"
            can_pass = False
        else:
            oi = fetch_object_info(args.url, args.timeout)
            problems, n_nodes, n_models, notes = check_online(oi, node_types, models, nodes)
            mode = "online via %s/object_info" % args.url.rstrip("/")
    except Unknown as exc:
        print("UNKNOWN - the check did not run: %s" % exc)
        print("This is not a pass. Resolve the cause and run again.")
        return UNKNOWN

    return report(problems, n_nodes, n_models, mode, can_pass=can_pass,
                  notes=notes, notes_block=args.require_inputs)


if __name__ == "__main__":
    sys.exit(main())
