#!/usr/bin/env python3
"""Opt-in sidecar contention research; uses already installed model files."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"n": len(values), "p50_ms": statistics.median(values) / 1e6 if values else None,
            "p95_ms": ordered[max(0, math.ceil(len(values) * .95) - 1)] / 1e6 if values else None,
            "sum_seconds": sum(values) / 1e9}


def summarize(raw: dict, tick_rate: int) -> dict:
    stages = {}
    for stage in sorted({r["stage"] for r in raw["observations"]}):
        rows = [r for r in raw["observations"] if r["stage"] == stage]
        stages[stage] = {"wait": distribution([r["wait_ns"] for r in rows if r["wait_ns"] is not None]),
                         "execution": distribution([r["execution_ns"] for r in rows])}
    workers = {}
    for kind in ["interactive", "batch"]:
        rows = [r for r in raw["rows"] if r["kind"] == kind]
        workers[kind] = {"latency": distribution([r["latency_ns"] for r in rows]),
                         "checkout": distribution([r["checkout_ns"] for r in rows])}
        if kind == "interactive":
            workers[kind]["first_partial"] = distribution([r["first_partial_ns"] for r in rows if r["first_partial_ns"] is not None])
            workers[kind]["stop_flush"] = distribution([r["stop_flush_ns"] for r in rows])
            workers[kind]["chunk"] = distribution([n for r in rows for n in r["chunk_ns"]])
    elapsed = raw["elapsed_ns"] / 1e9
    return {"case":raw["case"], "pool":raw["pool"], "stages":stages,"workers":workers,
            "cpu_seconds":raw["process_cpu_ticks"] / tick_rate,"elapsed_seconds":elapsed,
            "audio_seconds_per_wall_second":raw["audio_seconds"] / elapsed,
            "requests_per_second":len(raw["rows"]) / elapsed,
            "peak_sampled_rss_mib":max(r["rss_kib"] for r in raw["rows"]) / 1024,
            "after_load_rss_mib":raw["after_load_rss_kib"] / 1024,
            "load_ms":raw["load_ns"] / 1e6,
            "cold_batch_ms":raw["cold_batch"]["latency_ns"] / 1e6,
            "cold_interactive_ms":raw["cold_interactive"]["latency_ns"] / 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",nargs="+",choices=["none","vad","punctuation","speaker","all"],default=["none","vad","punctuation","speaker","all"])
    parser.add_argument("--pools",nargs="+",type=int,choices=[1,2,4],default=[1,2,4])
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    if not args.summarize_only:
        model_dir = Path.home()/".gigastt/models"
        names = ["v3_rnnt_encoder_int8.onnx","v3_rnnt_decoder.onnx","v3_rnnt_joint.onnx","v3_vocab.txt","punct/rupunct_small_int8.onnx","punct/tokenizer.json","punct/config.json","vad/silero_vad.onnx","wespeaker_resnet34.onnx"]
        source_files = ["crates/gigastt-core/src/sidecar_probe.rs", "crates/gigastt-core/src/sidecar_probe/experiment.rs", "crates/gigastt-core/src/vad/silero.rs", "crates/gigastt-core/src/punctuation/mod.rs", "crates/gigastt-core/src/inference/diarization.rs"]
        fixtures = [f"crates/gigastt/tests/fixtures/golos_{i:02}.wav" for i in range(3)]
        provenance = {"fixtures_sha256":{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in fixtures},"source_sha256":{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_files},"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"platform":platform.platform(),"rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),"profile":"debug","cpu_ticks_per_second":os.sysconf("SC_CLK_TCK"),"models_sha256":{n:hashlib.sha256((model_dir/n).read_bytes()).hexdigest() for n in names}}
        (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        for case in args.cases:
            for pool in args.pools:
                name = f"{case}-p{pool}"; print(f"Running {name}",flush=True)
                env = dict(os.environ,GIGASTT_SIDECAR_PROBE=str(output/f"{name}.json"),GIGASTT_SIDECAR_CASE=case,GIGASTT_SIDECAR_POOL=str(pool))
                with (output/f"{name}.log").open("w") as log:
                    subprocess.run(["cargo","test","-p","gigastt-core","--lib","benchmark_sidecar_contention","--","--ignored","--nocapture","--test-threads=1"],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    tick_rate = json.loads((output/"provenance.json").read_text())["cpu_ticks_per_second"]
    summaries = {p.stem:summarize(json.loads(p.read_text()),tick_rate) for p in sorted(output.glob("*-p[124].json"))}
    (output/"summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))

if __name__ == "__main__":
    main()
