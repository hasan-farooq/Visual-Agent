"""
Agentic CT Assistant (Educational) — Lazy Model Loading
========================================================

Model loading strategy (the core fix):
────────────────────────────────────────
  App starts
      │
      ├─ Preprocessed files exist on disk?
      │       YES → load Qwen now  (TS never needed again)
      │       NO  → do NOT load Qwen yet
      │
  User clicks "Preprocess CT"
      │       Qwen not in memory → TS gets 100% of VRAM
      │       TS runs, saves seg + df + meta to disk
      │       TS exits memory
      │       → load_qwen() called for the first time
      │
  User clicks "Run Agentic Analysis"
              load_qwen() is a no-op (already loaded)
              Qwen reads df.parquet from disk
              TS is never involved again

Result: Qwen and TotalSegmentator NEVER share VRAM under any code path.
"""

import os
import io
import json
import ast
import re
import hashlib
from threading import Thread
from typing import Iterable

import numpy as np
import pandas as pd
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import spaces
import gradio as gr

from transformers import (
    Qwen3_5ForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes


# ─────────────────────────────────────────────────────────────────────────────
# Device / dtype
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)

QWEN_MODEL_NAME = "Qwen/Qwen3.5-2B"
PREPROCESS_DIR  = "./ct_preprocessed"
os.makedirs(PREPROCESS_DIR, exist_ok=True)

print(f"🖥️  Device: {DEVICE}")
print(f"📂  Preprocessed cache: {os.path.abspath(PREPROCESS_DIR)}")


# ─────────────────────────────────────────────────────────────────────────────
# TotalSegmentator label map (117 structures, task='total')
# ─────────────────────────────────────────────────────────────────────────────
LABEL_NAMES = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "sacrum",
    26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4", 29: "vertebrae_L3",
    30: "vertebrae_L2", 31: "vertebrae_L1", 32: "vertebrae_T12", 33: "vertebrae_T11",
    34: "vertebrae_T10", 35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7",
    38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4", 41: "vertebrae_T3",
    42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7", 45: "vertebrae_C6",
    46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2",
    50: "vertebrae_C1", 51: "heart", 52: "aorta", 53: "pulmonary_vein",
    54: "brachiocephalic_trunk", 55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left", 62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right", 69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
    79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right", 84: "gluteus_minimus_left",
    85: "gluteus_minimus_right", 86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right", 90: "brain", 91: "skull",
    92: "rib_left_1",  93: "rib_left_2",  94: "rib_left_3",  95: "rib_left_4",
    96: "rib_left_5",  97: "rib_left_6",  98: "rib_left_7",  99: "rib_left_8",
    100: "rib_left_9", 101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7", 111: "rib_right_8",
    112: "rib_right_9", 113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages",
}
NUM_LABELS = 117
NAME_TO_ID = {v: k for k, v in LABEL_NAMES.items()}
LABEL_MENU = "\n".join(f"{lid}: {name}" for lid, name in LABEL_NAMES.items())


# ─────────────────────────────────────────────────────────────────────────────
# Lazy Qwen loader  — called only when actually needed
# ─────────────────────────────────────────────────────────────────────────────
QWEN_MODEL     = None
QWEN_PROCESSOR = None


def load_qwen():
    """
    Load Qwen into VRAM if not already loaded.
    Safe to call multiple times — subsequent calls are no-ops.
    Never called while TotalSegmentator could be running.
    """
    global QWEN_MODEL, QWEN_PROCESSOR
    if QWEN_MODEL is not None:
        return  # already loaded — nothing to do

    print("⏳  Loading Qwen into VRAM …")
    try:
        QWEN_MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=QWEN_DTYPE,
            device_map=DEVICE,
        ).eval()
        QWEN_PROCESSOR = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        print("✅  Qwen loaded — shared by all agent roles.")
    except Exception as e:
        print(f"❌  Qwen failed to load: {e}")
        QWEN_MODEL     = None
        QWEN_PROCESSOR = None
        raise


def qwen_is_ready() -> bool:
    return QWEN_MODEL is not None and QWEN_PROCESSOR is not None


# ─────────────────────────────────────────────────────────────────────────────
# Startup: load Qwen only if preprocessing already done
# ─────────────────────────────────────────────────────────────────────────────
def _any_preprocessed_on_disk() -> bool:
    """True if at least one complete preprocessed volume exists on disk."""
    try:
        return any(
            f.endswith("_df.parquet")
            for f in os.listdir(PREPROCESS_DIR)
        )
    except Exception:
        return False


if _any_preprocessed_on_disk():
    print("📂  Preprocessed volume(s) found on disk → loading Qwen at startup.")
    load_qwen()
else:
    print("⚠️   No preprocessed volumes found.")
    print("     Qwen will load AFTER TotalSegmentator finishes preprocessing.")
    print("     Upload a CT and click 'Preprocess CT' first.")


# ─────────────────────────────────────────────────────────────────────────────
# TotalSegmentator — lazy import
# ─────────────────────────────────────────────────────────────────────────────
try:
    from totalsegmentator.python_api import totalsegmentator as _totalsegmentator
    TS_AVAILABLE = True
    print("✅  TotalSegmentator import OK")
except Exception as e:
    _totalsegmentator = None
    TS_AVAILABLE      = False
    print(f"⚠️   TotalSegmentator not available: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Disk-based preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────
def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_preprocess_paths(md5: str) -> dict:
    base = os.path.join(PREPROCESS_DIR, md5)
    return {
        "seg":  f"{base}_seg.nii.gz",
        "df":   f"{base}_df.parquet",
        "meta": f"{base}_meta.json",
    }


def is_preprocessed(md5: str) -> bool:
    return all(os.path.exists(p) for p in get_preprocess_paths(md5).values())


def preprocess_ct(ct_path: str) -> dict:
    """
    STAGE 1 — Run TotalSegmentator (Qwen must NOT be in VRAM).

    Guarantee: this function is only ever called when Qwen is not loaded
    (fresh start) or after Qwen has been explicitly kept off GPU.
    After saving outputs to disk, it calls load_qwen() so the agentic
    pipeline is immediately ready.
    """
    if not TS_AVAILABLE:
        raise gr.Error("TotalSegmentator is not installed in this environment.")

    md5   = file_md5(ct_path)
    paths = get_preprocess_paths(md5)

    if is_preprocessed(md5):
        print(f"📂  Already preprocessed ({md5[:8]}…) — skipping TS.")
        # make sure Qwen is ready even if this is a repeat call
        load_qwen()
        return paths

    # Safety: if Qwen somehow ended up loaded, evict it before TS starts.
    # Under normal flow this branch is never hit.
    global QWEN_MODEL, QWEN_PROCESSOR
    if QWEN_MODEL is not None:
        print("⚠️   Qwen was in VRAM before TS — evicting now.")
        QWEN_MODEL.to("cpu")
        QWEN_MODEL     = None
        QWEN_PROCESSOR = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"🩻  Running TotalSegmentator ({md5[:8]}…) with full VRAM …")

    try:
        input_img = nib.load(ct_path)
        zooms     = tuple(float(z) for z in input_img.header.get_zooms()[:3])

        # fast=True → 3 mm model, ~6 GB peak VRAM, ~45 s for a 511×404×339 volume
        seg_img = _totalsegmentator(
            input=input_img,
            ml=True,    # single multilabel volume (values 1..117)
            fast=True,  # 3 mm — sufficient for anatomy education
            quiet=True,
        )

        # save segmentation volume
        nib.save(seg_img, paths["seg"])

        # build + save structure dataframe
        seg_arr = np.asarray(seg_img.dataobj).astype(np.uint8)
        df      = build_structure_dataframe(seg_arr, zooms)
        df.to_parquet(paths["df"], index=False)

        # save metadata
        with open(paths["meta"], "w") as f:
            json.dump({"md5": md5, "zooms": list(zooms), "shape": list(seg_arr.shape)}, f)

        print(f"✅  TS done — outputs saved to {PREPROCESS_DIR}/")

    except Exception:
        # clean up partial files so is_preprocessed() stays consistent
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
        raise

    finally:
        # TS is done and out of memory — now safe to load Qwen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # load Qwen now that VRAM is free
    load_qwen()
    return paths


def load_df_and_meta(md5: str) -> tuple[pd.DataFrame, dict]:
    """Read parquet + JSON from disk. TS not involved. Sub-100 ms."""
    paths = get_preprocess_paths(md5)
    df    = pd.read_parquet(paths["df"])
    with open(paths["meta"]) as f:
        meta = json.load(f)
    return df, meta


def load_arrays_for_preview(ct_path: str, md5: str):
    """Load CT (int16) + saved segmentation (uint8) for overlay rendering."""
    paths   = get_preprocess_paths(md5)
    ct_img  = nib.load(ct_path)
    seg_img = nib.load(paths["seg"])
    ct_arr  = np.rint(np.asarray(ct_img.dataobj)).astype(np.int16)
    seg_arr = np.asarray(seg_img.dataobj).astype(np.uint8)
    zooms   = tuple(float(z) for z in ct_img.header.get_zooms()[:3])
    return ct_arr, seg_arr, zooms


# ─────────────────────────────────────────────────────────────────────────────
# Structure dataframe builder
# ─────────────────────────────────────────────────────────────────────────────
def build_structure_dataframe(seg_arr: np.ndarray, zooms) -> pd.DataFrame:
    voxel_vol_mm3 = float(np.prod(zooms[:3])) if len(zooms) >= 3 else 1.0
    counts = np.bincount(seg_arr.ravel(), minlength=NUM_LABELS + 1)
    rows   = []
    for lid in range(1, NUM_LABELS + 1):
        name = LABEL_NAMES.get(lid, f"label_{lid}")
        vox  = int(counts[lid])
        if vox == 0:
            rows.append({
                "id": lid, "structure": name, "present": False,
                "voxels": 0, "volume_ml": 0.0,
                "z_min": None, "z_max": None, "z_mid": None,
                "centroid_x": None, "centroid_y": None, "centroid_z": None,
            })
            continue
        coords = np.argwhere(seg_arr == lid)
        z_vals = coords[:, 2]
        rows.append({
            "id":         lid,
            "structure":  name,
            "present":    True,
            "voxels":     vox,
            "volume_ml":  round(vox * voxel_vol_mm3 / 1000.0, 2),
            "z_min":      int(z_vals.min()),
            "z_max":      int(z_vals.max()),
            "z_mid":      int(round(float(z_vals.mean()))),
            "centroid_x": int(round(float(coords[:, 0].mean()))),
            "centroid_y": int(round(float(coords[:, 1].mean()))),
            "centroid_z": int(round(float(z_vals.mean()))),
        })
    return pd.DataFrame(rows)


def summarize_df_for_qwen(df: pd.DataFrame, requested_ids: list) -> dict:
    requested = []
    for lid in requested_ids:
        row = df[df["id"] == lid]
        if row.empty:
            continue
        r = row.iloc[0]
        requested.append({
            "id":          int(r["id"]),
            "structure":   r["structure"],
            "found":       bool(r["present"]),
            "volume_ml":   float(r["volume_ml"]),
            "slice_range": ([int(r["z_min"]), int(r["z_max"])] if r["present"] else None),
        })
    other_present = (
        df[(df["present"]) & (~df["id"].isin(requested_ids))]["structure"].tolist()
    )
    return {
        "requested_structures":             requested,
        "num_requested_found":              sum(1 for x in requested if x["found"]),
        "num_requested_missing":            sum(1 for x in requested if not x["found"]),
        "other_structures_present_in_scan": other_present,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Overlay preview
# ─────────────────────────────────────────────────────────────────────────────
def _tab20(n):
    try:
        return matplotlib.colormaps["tab20"].resampled(n)
    except Exception:
        return plt.cm.get_cmap("tab20", n)


def render_overlay_preview(ct_arr, seg_arr, target_ids) -> Image.Image:
    target_ids = [t for t in target_ids if 1 <= t <= NUM_LABELS]
    union = np.isin(seg_arr, target_ids) if target_ids else seg_arr > 0

    zs = np.where(union.any(axis=(0, 1)))[0]
    if zs.size == 0:
        zs = np.where((seg_arr > 0).any(axis=(0, 1)))[0]
    zsel = (
        [ct_arr.shape[2] // 2] if zs.size == 0
        else sorted({int(np.quantile(zs, q)) for q in (0.25, 0.5, 0.75)})
    )

    cmap = _tab20(NUM_LABELS + 1)
    fig, axes = plt.subplots(1, len(zsel), figsize=(4 * len(zsel), 4.2))
    if len(zsel) == 1:
        axes = [axes]

    for ax, z in zip(axes, zsel):
        ax.imshow(ct_arr[:, :, z].T, cmap="gray", origin="lower", vmin=-200, vmax=300)
        seg_slice = seg_arr[:, :, z].T
        rgba      = cmap(seg_slice / NUM_LABELS)
        show      = np.isin(seg_slice, target_ids) if target_ids else seg_slice > 0
        rgba[..., 3] = np.where(show, 0.5, 0.0)
        ax.imshow(rgba, origin="lower")
        ax.set_title(f"slice {z}", fontsize=9, color="white")
        ax.axis("off")

    fig.patch.set_facecolor("black")
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def safe_parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    m    = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(candidate)
    except Exception:
        return {}


def safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else max(0.0, min(1.0, v))
    except Exception:
        return default


def coerce_label_ids(raw_ids, raw_names=None) -> list:
    out = []
    for v in (raw_ids or []):
        try:
            iv = int(v)
            if 1 <= iv <= NUM_LABELS:
                out.append(iv)
        except Exception:
            name = str(v).strip().lower()
            if name in NAME_TO_ID:
                out.append(NAME_TO_ID[name])
    for n in (raw_names or []):
        name = str(n).strip().lower()
        if name in NAME_TO_ID:
            out.append(NAME_TO_ID[name])
    return list(dict.fromkeys(out))


# ─────────────────────────────────────────────────────────────────────────────
# Qwen helpers  (all guard on qwen_is_ready())
# ─────────────────────────────────────────────────────────────────────────────
def _build_text_inputs(prompt_text: str):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    chat = QWEN_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        return QWEN_PROCESSOR(text=[chat], return_tensors="pt", padding=True).to(QWEN_MODEL.device)
    except Exception:
        return QWEN_PROCESSOR.tokenizer([chat], return_tensors="pt", padding=True).to(QWEN_MODEL.device)


def qwen_json(instruction: str, max_tokens: int = 700, temperature: float = 0.15) -> dict:
    if not qwen_is_ready():
        raise gr.Error("Qwen is not loaded. Please preprocess a CT volume first.")
    inputs = _build_text_inputs(instruction)
    with torch.inference_mode():
        gen_ids = QWEN_MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    raw = QWEN_PROCESSOR.batch_decode(
        gen_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()
    del inputs, gen_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parsed = safe_parse_json(raw)
    return parsed if isinstance(parsed, dict) else {}


def qwen_stream(instruction: str, max_tokens: int = 950, temperature: float = 0.4):
    if not qwen_is_ready():
        raise gr.Error("Qwen is not loaded. Please preprocess a CT volume first.")
    inputs   = _build_text_inputs(instruction)
    streamer = TextIteratorStreamer(
        QWEN_PROCESSOR.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180,
    )
    Thread(
        target=QWEN_MODEL.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            use_cache=True,
            temperature=temperature,
            do_sample=True,
        ),
    ).start()
    acc = ""
    for tok in streamer:
        acc += tok
        yield acc
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Agent steps  (Qwen reads df only — TS completely absent)
# ─────────────────────────────────────────────────────────────────────────────
def planner_step(user_query: str, feedback: str | None, already_selected: list, iteration: int) -> dict:
    feedback_text = feedback or "No previous self-reflection feedback."
    selected_text = (
        ", ".join(f"{i}-{LABEL_NAMES[i]}" for i in already_selected)
        if already_selected else "none yet"
    )
    instruction = f"""
You are the PLANNER & TASK-ASSIGNER in an educational CT anatomy assistant.

Available structures (TotalSegmentator labels — use ONLY these ids):
{LABEL_MENU}

User question:
"{user_query}"

Already selected so far: {selected_text}
Self-reflection feedback from previous iteration:
"{feedback_text}"
Iteration: {iteration}

Your task:
Use clinical/anatomical reasoning to decide which structures are needed
to teach the user about their question, and what they should focus on.

Return ONLY valid JSON in this exact format:
{{
  "target_label_ids": [5, 2, 3],
  "target_names": ["liver", "kidney_right", "kidney_left"],
  "clinical_reasoning": "why these structures answer the question",
  "educational_focus": "what the learner should look for / focus on",
  "plan_goal": "goal for this iteration"
}}

Rules:
- Pick the SMALLEST set of structures that fully covers the question.
- If previous feedback asked for more structures, ADD them (keep old ones).
- Use only ids/names from the list above.
- Educational only — do NOT diagnose, do NOT invent findings.
- Return JSON only, no markdown.
"""
    result   = qwen_json(instruction, max_tokens=650, temperature=0.1)
    ids      = coerce_label_ids(result.get("target_label_ids"), result.get("target_names"))
    ids      = list(dict.fromkeys(list(already_selected) + ids))
    fallback = not ids

    return {
        "target_label_ids":   ids,
        "target_names":       [LABEL_NAMES[i] for i in ids],
        "clinical_reasoning": str(result.get("clinical_reasoning", "") or "").strip(),
        "educational_focus":  str(result.get("educational_focus",  "") or "").strip(),
        "plan_goal":          str(result.get("plan_goal", f"Iteration {iteration} plan") or "").strip(),
        "fallback_used":      fallback,
    }


def reflection_step(user_query: str, plan: dict, df_summary: dict, iteration: int, max_iter: int) -> dict:
    instruction = f"""
You are the SELF-REFLECTION agent in an educational CT anatomy assistant.

User question:
"{user_query}"

Planner's educational focus:
"{plan.get('educational_focus', '')}"

Segmentation results for requested structures:
{json.dumps(df_summary, indent=2)}

Iteration {iteration} of at most {max_iter}.

Decide whether the structures are COMPLETE and CLINICALLY SUFFICIENT to give
a good educational answer.

Return ONLY valid JSON in this exact format:
{{
  "sufficient": true,
  "confidence": 0.85,
  "critique": "what is good or missing",
  "missing_label_ids": [],
  "feedback_to_planner": "if not sufficient, exactly which structures to add and why"
}}

Decide NOT sufficient if:
- A requested structure relevant to the question was NOT found in the scan, OR
- An obviously related neighbouring structure should also be included.

Rules:
- missing_label_ids must come only from structures present in the scan.
- Be reasonable — do not demand the whole body for a focused question.
- confidence between 0 and 1. Return JSON only.
"""
    result         = qwen_json(instruction, max_tokens=600, temperature=0.1)
    sufficient_raw = result.get("sufficient", True)
    if isinstance(sufficient_raw, str):
        sufficient = sufficient_raw.strip().lower() in {"true", "yes", "sufficient", "pass"}
    else:
        sufficient = bool(sufficient_raw)

    return {
        "sufficient":          sufficient,
        "confidence":          safe_float(result.get("confidence", 0.0)),
        "critique":            str(result.get("critique",            "") or "").strip(),
        "missing_label_ids":   coerce_label_ids(result.get("missing_label_ids")),
        "feedback_to_planner": str(result.get("feedback_to_planner", "") or "").strip(),
    }


def final_answer_instruction(user_query: str, plan: dict, df_summary: dict) -> str:
    return f"""
You are an anatomy & radiology TEACHER. Write a clear educational answer for a
student/clinician based ONLY on the CT segmentation facts below.

User question:
"{user_query}"

Educational focus:
"{plan.get('educational_focus', '')}"

Segmentation facts (volumes in ml, slice ranges are axial indices):
{json.dumps(df_summary, indent=2)}

Write a well-structured answer with these headings:

1. **What was segmented** — structures found, their size, axial position.
2. **What to look for** — shape, symmetry, size, edges, neighbour relationships.
3. **Anatomy & teaching context** — why these structures matter for the question.
4. **Limitations** — automatic model caveats; this is educational, NOT a diagnosis.

Rules:
- Use ONLY the provided facts. Do not invent measurements or pathology.
- Do not give a diagnosis.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 Gradio handler — "Preprocess CT" button
# ─────────────────────────────────────────────────────────────────────────────
@spaces.GPU(duration=300)
def run_preprocessing(ct_file):
    """
    Runs TotalSegmentator on the uploaded CT.
    Qwen is NOT in VRAM at this point (either never loaded, or evicted).
    After TS finishes, load_qwen() is called inside preprocess_ct().
    """
    if not TS_AVAILABLE:
        return "❌ TotalSegmentator is not available in this environment."
    if ct_file is None:
        return "⚠️ Please upload a CT volume first."

    ct_path = ct_file if isinstance(ct_file, str) else ct_file.name
    md5     = file_md5(ct_path)

    if is_preprocessed(md5):
        df, meta  = load_df_and_meta(md5)
        n_present = int(df["present"].sum())
        # Still call load_qwen in case this is the first session with cached data
        load_qwen()
        return (
            f"📂 Already preprocessed ({md5[:8]}…)\n"
            f"   Shape: {meta['shape']}  |  Structures found: {n_present}/{NUM_LABELS}\n"
            f"   Qwen ready: {qwen_is_ready()}"
        )

    try:
        preprocess_ct(ct_path)   # TS runs here; load_qwen() called inside
        df, meta  = load_df_and_meta(md5)
        n_present = int(df["present"].sum())
        return (
            f"✅ Preprocessing complete ({md5[:8]}…)\n"
            f"   Shape: {meta['shape']}  |  Structures found: {n_present}/{NUM_LABELS}\n"
            f"   Qwen loaded and ready."
        )
    except Exception as e:
        return f"❌ Preprocessing failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 Gradio handler — "Run Agentic Analysis" button
# ─────────────────────────────────────────────────────────────────────────────
EMPTY_DF = pd.DataFrame(columns=["id", "structure", "present", "voxels",
                                  "volume_ml", "z_min", "z_max", "z_mid"])


@spaces.GPU(duration=600)
def run_agentic_ct(ct_file, user_query, max_iterations):
    if ct_file is None:
        raise gr.Error("Please upload a CT volume.")
    if not user_query or not user_query.strip():
        raise gr.Error("Please enter a clinical / educational question.")

    ct_path        = ct_file if isinstance(ct_file, str) else ct_file.name
    md5            = file_md5(ct_path)
    max_iterations = int(max(1, min(3, int(max_iterations))))

    status       = ""
    answer       = ""
    plan_text    = ""
    reflect_text = ""
    trace_text   = ""
    preview      = None
    df_display   = EMPTY_DF

    def snapshot():
        return (status, answer, df_display, preview, plan_text, reflect_text, trace_text)

    # ── Guard: must be preprocessed before Qwen can work ─────────────────────
    if not is_preprocessed(md5):
        if not TS_AVAILABLE:
            raise gr.Error(
                "This CT has not been preprocessed and TotalSegmentator is unavailable. "
                "Please click 'Preprocess CT' first."
            )
        status = "🩻 Not preprocessed yet — running TotalSegmentator (one-time, ~45 s) …"
        yield snapshot()
        preprocess_ct(ct_path)  # TS runs; load_qwen() called at the end
    else:
        status = "📂 Segmentation found on disk — loading …"
        yield snapshot()
        load_qwen()  # no-op if already loaded

    if not qwen_is_ready():
        raise gr.Error("Qwen failed to load. Check logs.")

    # ── Load df + arrays ──────────────────────────────────────────────────────
    df, meta       = load_df_and_meta(md5)
    df_display     = df[df["present"]].reset_index(drop=True)
    ct_arr, seg_arr, zooms = load_arrays_for_preview(ct_path, md5)

    status = f"✅ {int(df['present'].sum())} structures available. Qwen planning …"
    yield snapshot()

    logs             = []
    selected_ids     = []
    feedback         = None
    final_plan       = None
    final_reflection = None

    # ── Agentic loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        status = f"🧠 Planner — iteration {iteration}/{max_iterations} …"
        yield snapshot()

        plan = planner_step(user_query, feedback, selected_ids, iteration)
        if plan["fallback_used"]:
            plan["target_label_ids"] = df[df["present"]]["id"].astype(int).tolist()
            plan["target_names"]     = [LABEL_NAMES[i] for i in plan["target_label_ids"]]

        selected_ids = plan["target_label_ids"]
        final_plan   = plan
        plan_text    = json.dumps(plan, indent=2)

        status = f"🔬 Surfacing {len(selected_ids)} structure(s) from saved segmentation …"
        yield snapshot()

        df_display = df[df["id"].isin(selected_ids)].reset_index(drop=True)
        preview    = render_overlay_preview(ct_arr, seg_arr, selected_ids)
        yield snapshot()

        status = f"🔍 Self-reflection — iteration {iteration} sufficient?"
        yield snapshot()

        df_summary       = summarize_df_for_qwen(df, selected_ids)
        reflection       = reflection_step(user_query, plan, df_summary, iteration, max_iterations)
        final_reflection = reflection
        reflect_text     = json.dumps(reflection, indent=2)

        logs.append({
            "iteration":            iteration,
            "planner":              plan,
            "segmentation_summary": df_summary,
            "self_reflection":      reflection,
        })
        trace_text = json.dumps(logs, indent=2)
        yield snapshot()

        if reflection["sufficient"]:
            status = f"✅ Sufficient after {iteration} iteration(s). Writing answer …"
            yield snapshot()
            break

        miss       = reflection.get("missing_label_ids", [])
        miss_names = ", ".join(LABEL_NAMES[i] for i in miss) if miss else "(none specified)"
        feedback   = (
            f"{reflection.get('feedback_to_planner', '')} "
            f"Add these structures: {miss_names}."
        )
        selected_ids = list(dict.fromkeys(selected_ids + miss))
        status = f"↩️ Not sufficient — re-planning (iteration {iteration + 1}) …"
        yield snapshot()
    else:
        status = "⛔ Max iterations reached. Writing best available answer …"
        yield snapshot()

    # ── Final streamed answer ─────────────────────────────────────────────────
    df_summary  = summarize_df_for_qwen(df, selected_ids)
    instruction = final_answer_instruction(user_query, final_plan, df_summary)
    for partial in qwen_stream(instruction, max_tokens=950, temperature=0.4):
        answer = partial
        yield snapshot()

    conf   = final_reflection.get("confidence", 0.0) if final_reflection else 0.0
    status = (
        f"Done — iterations: {len(logs)} | structures: {len(selected_ids)} | "
        f"sufficiency: {final_reflection.get('sufficient') if final_reflection else 'n/a'} "
        f"(conf {conf:.2f})"
    )
    yield snapshot()


# ─────────────────────────────────────────────────────────────────────────────
# Theme + CSS
# ─────────────────────────────────────────────────────────────────────────────
colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8", c100="#D3E5F0", c200="#A8CCE1", c300="#7DB3D2",
    c400="#529AC3", c500="#4682B4", c600="#3E72A0", c700="#36638C",
    c800="#2E5378", c900="#264364", c950="#1E3450",
)


class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue:   colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue:   colors.Color | str = colors.slate,
        text_size:     sizes.Size   | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue,
            neutral_hue=neutral_hue, text_size=text_size,
            font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            slider_color="*secondary_500",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
        )


steel_blue_theme = SteelBlueTheme()

css = r"""
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
body, .gradio-container { font-family: 'Outfit', sans-serif !important; }
footer { display: none !important; }
.app-header {
    background: linear-gradient(135deg, #1E3450 0%, #264364 30%, #3E72A0 70%, #4682B4 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 22px; color: #fff;
    box-shadow: 0 8px 32px rgba(30,52,80,0.25);
}
.app-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing:-0.02em; }
.app-header p  { margin: 0; color: rgba(255,255,255,0.82); font-size: 0.95rem; }
.flow {
    display:flex; flex-wrap:wrap; gap:8px; margin-top:14px;
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
}
.flow span {
    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.15);
    padding:4px 10px; border-radius:20px; color:#fff;
}
.stage-box { border-radius:10px; padding:12px 16px; margin-bottom:12px; font-size:0.9rem; }
.stage-1   { background:rgba(70,130,180,0.07); border-left:4px solid #4682B4; }
.stage-2   { background:rgba(29,158,117,0.07); border-left:4px solid #1D9E75; }
.dark .stage-1 { background:rgba(70,130,180,0.12); }
.dark .stage-2 { background:rgba(29,158,117,0.12); }
.disclaimer {
    background:rgba(70,130,180,0.06); border-left:4px solid #4682B4;
    border-radius:8px; padding:12px 16px; margin-bottom:16px;
    font-size:0.9rem; color:#2E5378;
}
.dark .disclaimer { color:#A8CCE1; }
.gradio-textbox textarea { font-family:'IBM Plex Mono',monospace !important; font-size:0.9rem !important; }
label { font-weight:600 !important; }
"""


def html_header():
    return """
    <div class="app-header">
        <h1>🩻 Agentic CT Tutor</h1>
        <p>
            Stage 1: TotalSegmentator preprocesses the CT once (Qwen not loaded).
            Stage 2: Qwen loads after TS exits — reads saved data, reasons, teaches.
            The two models never share VRAM.
        </p>
        <div class="flow">
            <span>Upload CT</span>
            <span>→ Preprocess (TS only)</span>
            <span>→ Qwen loads</span>
            <span>→ Plan → Reflect (≤3×)</span>
            <span>→ Answer</span>
        </div>
    </div>
    """


EXAMPLES = [
    ["Show me the kidneys and tell me what to check for cysts."],
    ["I want to learn about the liver and nearby organs on this scan."],
    ["Explain the thoracic vertebrae and ribs visible here."],
    ["What should I focus on when assessing the lungs?"],
]


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=steel_blue_theme, css=css, title="Agentic CT Tutor") as demo:
    gr.HTML(html_header())
    gr.HTML(
        '<div class="disclaimer"><strong>Educational tool.</strong> Outputs come from an '
        'automatic segmentation model and an LLM. This is for learning about anatomy on a CT '
        '— it is <strong>not a diagnosis</strong>. A qualified clinician must interpret real scans.</div>'
    )

    with gr.Row():

        # ── Left: inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML(
                '<div class="stage-box stage-1"><strong>Stage 1 — Preprocess CT</strong><br>'
                'Qwen is not loaded yet. TotalSegmentator gets the full GPU. '
                'Results saved to disk permanently. Qwen loads after TS finishes.</div>'
            )
            ct_input          = gr.File(label="CT volume (.nii / .nii.gz)",
                                        file_types=[".nii", ".gz"], type="filepath")
            preprocess_btn    = gr.Button("Preprocess CT  (run TotalSegmentator)", variant="secondary")
            preprocess_status = gr.Textbox(label="Preprocessing status", interactive=False, lines=3)

            gr.HTML(
                '<div class="stage-box stage-2" style="margin-top:16px">'
                '<strong>Stage 2 — Ask Qwen</strong><br>'
                'Qwen reads the saved segmentation dataframe. TotalSegmentator is never touched again.</div>'
            )
            query_input = gr.Textbox(
                label="Clinical / educational question",
                placeholder="e.g. Show me the kidneys and what to look for regarding cysts",
                lines=3,
            )
            with gr.Accordion("Settings", open=False):
                max_iter_slider = gr.Slider(minimum=1, maximum=3, value=3, step=1,
                                            label="Max planner / self-reflection iterations")
            run_btn    = gr.Button("Run Agentic Analysis  (Qwen only)", variant="primary")
            gr.Examples(examples=EXAMPLES, inputs=[query_input], label="Example questions")
            status_box = gr.Textbox(label="Pipeline status", interactive=False, lines=2)

        # ── Middle: preview + df ──────────────────────────────────────────────
        with gr.Column(scale=1):
            preview_img = gr.Image(label="Segmentation overlay (requested structures)", height=320)
            df_view     = gr.Dataframe(label="Structure dataframe (what Qwen reasons over)",
                                       interactive=False, wrap=True)

        # ── Right: answer + trace ─────────────────────────────────────────────
        with gr.Column(scale=1):
            answer_box = gr.Markdown(label="Educational answer")
            with gr.Accordion("Agent trace", open=False):
                plan_box    = gr.Textbox(label="Planner output",        lines=10, interactive=False)
                reflect_box = gr.Textbox(label="Self-reflection output", lines=8,  interactive=False)
                trace_box   = gr.Textbox(label="Full iteration trace",  lines=16, interactive=False)

    preprocess_btn.click(
        fn=run_preprocessing,
        inputs=[ct_input],
        outputs=[preprocess_status],
    )
    run_btn.click(
        fn=run_agentic_ct,
        inputs=[ct_input, query_input, max_iter_slider],
        outputs=[status_box, answer_box, df_view, preview_img,
                 plan_box, reflect_box, trace_box],
    )


if __name__ == "__main__":
    demo.launch(show_error=True, ssr_mode=False, share=True)

