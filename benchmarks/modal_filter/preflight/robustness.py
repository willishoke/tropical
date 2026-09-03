#!/usr/bin/env python3
"""Is the ~393 instr/mode/sample slope a property of the composition, or of one
pole geometry?  Re-fit the filter's marginal audio cost at three more configs:
a high cutoff, a high resonance, and a long decay (which moves the SOURCE
poles).  If the slope is geometry-dependent, the preflight's linear model is an
artifact and everything downstream of it is suspect."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import marginal

CONFIGS = [
    ("baseline",  {"cutoff": 800,  "resonance": 0.5}, {"freq": 220, "decay": 4}),
    ("hi-cutoff", {"cutoff": 2500, "resonance": 0.5}, {"freq": 220, "decay": 4}),
    ("hi-res",    {"cutoff": 800,  "resonance": 0.9}, {"freq": 220, "decay": 4}),
    ("long-dec",  {"cutoff": 800,  "resonance": 0.5}, {"freq": 220, "decay": 12}),
]

if __name__ == "__main__":
    root_work = marginal.WORK.parent / ".work-robust"
    print(f"{'config':>10} {'d16':>7} {'d64':>7} {'slope':>7} {'icept':>7} {'d_ns%64':>8}")
    for name, fparams, rparams in CONFIGS:
        marginal.WORK = root_work / name
        marginal.WORK.mkdir(parents=True, exist_ok=True)
        marginal.STAGES["filter"]["params"] = fparams
        base_params = {"freq": rparams["freq"], "decay": rparams["decay"]}
        orig = marginal.graph
        def patched(partials, stage, _bp=base_params, _orig=orig):
            g = _orig(partials, stage)
            g["nodes"][0]["params"].update(_bp)
            return g
        marginal.graph = patched
        try:
            b16, f16 = marginal.run(16, None), marginal.run(16, "filter")
            b64, f64 = marginal.run(64, None), marginal.run(64, "filter")
        finally:
            marginal.graph = orig
        d16 = f16["audio"] - b16["audio"]
        d64 = f64["audio"] - b64["audio"]
        slope = (d64 - d16) / 48.0
        icept = d64 - slope * 64
        dns = 100.0 * (f64["ns"] - b64["ns"]) / b64["ns"]
        print(f"{name:>10} {d16:>7} {d64:>7} {slope:>7.1f} {icept:>7.0f} {dns:>7.1f}%",
              flush=True)
