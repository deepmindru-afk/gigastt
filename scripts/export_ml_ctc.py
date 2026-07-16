#!/usr/bin/env python3
"""Export ai-sage/GigaAM-Multilingual (charwise-CTC) to a single ONNX for gigastt.

The multilingual CTC model shares GigaAM v3's frontend (torchaudio MelSpectrogram:
64 mel, n_fft=320, win=320, hop=160, center=false, natural-log) and Conformer-768
encoder, so gigastt's existing `features.rs` feeds it unchanged. The CTC head is a
single self-contained ONNX (no separate decoder/joiner like RNN-T).

Outputs (in --out):
  multilingual_ctc.onnx   features[B,64,T] + feature_lengths[B]
                          -> log_probs[B,T',71] + encoded_lengths[B]   (opset 17, fp32)
  multilingual_ctc.yaml   model cfg (emitted by to_onnx)
  ml_ctc_vocab.txt        the 70-token CTC vocabulary, one per line; CTC blank id = 70
  golden.npz              (features, log_probs) reference for validating the Rust CTC head

Env (local Python 3.14 is broken -> use uv + python3.13):
  uv venv --python 3.13 .venv-gigaam
  uv pip install --python .venv-gigaam/bin/python \
      torch torchaudio transformers huggingface_hub hydra-core omegaconf \
      soundfile sentencepiece numpy onnx onnxruntime
  HF_HUB_DISABLE_XET=1 .venv-gigaam/bin/python scripts/export_ml_ctc.py --out onnx_ml_ctc

Reproduces the export path from the model's own `modeling_gigaam.py::GigaAMASR._to_onnx`
(the "ctc" branch): swaps forward -> forward_for_export and traces encoder+CTCHead.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

# The GigaAM modeling file lazily imports optional heavy deps — pyannote.audio
# (VAD/diarization) and flash_attn/einops (flash attention) — inside functions a
# CTC-encoder export never calls (flash attention is off; config `flash_attn: false`).
# transformers' static `check_imports` would still demand all of them, and flash_attn
# cannot even build on macOS. Neutralize the check: the real imports stay lazy and are
# never triggered by `from_pretrained` + `to_onnx`.
import transformers.dynamic_module_utils as _dmu  # noqa: E402

_dmu.check_imports = lambda *_a, **_k: []

REPO = "ai-sage/GigaAM-Multilingual"
REVISION = "ctc"  # == main; the CTC ASR checkpoint (220M). Use "large_ctc" for the 600M.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="onnx_ml_ctc", help="output directory")
    ap.add_argument("--revision", default=REVISION, help="ctc | large_ctc")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading {REPO}@{args.revision} (trust_remote_code) ...")
    model = AutoModel.from_pretrained(REPO, revision=args.revision, trust_remote_code=True)
    model.eval()
    # The HF `GigaAMModel` wrapper holds the real GigaAMASR at `.model`; cfg / encoder /
    # decoding / forward_for_export live there. Its `to_onnx` takes only `dir_path`.
    inner = model.model

    # --- vocab (CTC blank id = len(vocab)) ---
    vocab = list(inner.cfg.decoding.vocabulary)
    (out / "ml_ctc_vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"[2/4] wrote ml_ctc_vocab.txt: {len(vocab)} tokens, blank_id={len(vocab)}")

    # --- ONNX export via the model's own to_onnx() ---
    print("[3/4] exporting ONNX via model.to_onnx() ...")
    model.to_onnx(str(out))
    onnx_path = out / f"{inner.cfg.model_name}.onnx"
    print(f"      -> {onnx_path}")

    # --- golden-reference numerical equivalence (PyTorch vs onnxruntime) ---
    print("[4/4] golden-reference check (PyTorch forward_for_export vs onnxruntime) ...")
    import onnxruntime as ort

    feats, flens = inner.encoder.input_example()
    # to_onnx traces under encoder.onnx_export_mode() (flash/SDPA attention disabled),
    # so the reference MUST run in the same mode or the logits diverge (this bit us once).
    with torch.no_grad(), inner.encoder.onnx_export_mode():
        log_probs_pt, _enc_len_pt = inner.forward_for_export(feats, flens)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    log_probs_onnx = sess.run(
        None,
        {"features": feats.numpy().astype(np.float32), "feature_lengths": flens.numpy()},
    )[0]
    lp_pt = log_probs_pt.numpy()
    delta = float(np.max(np.abs(lp_pt - log_probs_onnx)))
    # For CTC greedy decoding the per-frame argmax is what matters; require an exact match.
    argmax_agree = float((lp_pt.argmax(-1) == log_probs_onnx.argmax(-1)).mean())
    print(f"      features={tuple(feats.shape)} log_probs={tuple(log_probs_onnx.shape)}")
    print(f"      max|Δ log_probs| = {delta:.3e}   argmax agreement = {argmax_agree:.4f}")
    ok = delta < 1e-2 and argmax_agree == 1.0
    print(f"      -> {'OK' if ok else 'FAIL'}")

    np.savez(out / "golden.npz", features=feats.numpy(), log_probs=lp_pt)
    print(f"      wrote {out / 'golden.npz'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
