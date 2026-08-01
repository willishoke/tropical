# Demo release decision log

- Baseline: `91ecf0c8a590a416b25d5165dffc9d66a78aad68`
- Required implementation predecessor: `15e5a39bfce9dd242db59aa6ebc2460987aa2199`
- Handoff ref: `demo/room-position-handoff-2026-08-01`
- Release branch: `demo/modal-pocket-scene`
- Scope: the exact modal-pocket demo on the recorded release Mac; no general
  live-Metal support claim.

| ID | Status | Decision | Basis / consequence |
|---|---|---|---|
| DR-01 | frozen | Qualify one exact scene only. | The general-support boundary in the master remains unchanged. |
| DR-02 | implemented | Controls outrank scopes through independent RPC lanes, a counted runtime waiter signal, cooperative per-coordinate abort, and renderer freeze-hold. | A second socket alone cannot bound a scope already holding the runtime lock. |
| DR-03 | implemented | Preserve first, one reversal point, and latest/final value per parameter while an epoch is in flight. | Bounds the queue without erasing an out-and-back gesture. |
| DR-04 | superseded | Do not integrate Foundry 24, Industrial Cathedral 32, or Tanker 48. Use the frozen Mutable Clouds reference to qualify their replacement. | Listening found all three sparse stationary banks object-like, damp, and non-spatial. DR-12 now selects the replacement. |
| DR-05 | implemented | Primary hardware configuration is `Bdev=128`, `Rgpu=512`. | Independent-quantum Metal oracle and exact-scene diagnostics pass; no source-level universal default is claimed. |
| DR-06 | implemented | Use whole-signal worker morphing for the first 512-frame candidate tile, with a smoothstep old/new crossfade and pure-new second tile. | Hardware and offline oracles pass; the demo-local dual-filter fallback was not selected. |
| DR-07 | implemented | A frame-zero activation remains legal only while no callback has entered; exact-target publication is missed unless already claimed. | Preserves offline `set_sample_index` semantics while closing both boot and exact-`E` descriptor races. |
| DR-08 | implemented | Select a 384-point, 24 fps two-trace scope profile. | The provisional 512-point run measured 20.83 ms p99 against the 20 ms gate. The selected profile measured 12.08 ms p99 and 23.72 fps in the tightened short smoke; final-scene qualification remains required. |
| DR-09 | implemented | Qualification-only DAC capture wraps the existing one-buffer callback-safe state machine and is disabled unless `TROPICAL_QUALIFICATION=1`. | Produces exact next-callback evidence without adding a streaming product API or callback allocation. |
| DR-10 | implemented for audition | Freeze official Mutable Instruments Clouds commit `08460a69a7e1f7a81c5a2abcc7189c9a6b7208d4` as the room gold standard; render its native stereo output and mono fold-down at the exact demo source. | Gives the room lane a reproducible perceptual and quantitative target. The stateful reference is not linked into the product. |
| DR-11 | enabling result | Preserve complete delay pole lattices as grouped periodic carriers and analytically convolve them with fixed modal sources, instead of pruning or independently rendering every pole. | A 9,133-mode teacher was recovered through 12 groups × 12 source rows with `2.72e-13` relative error and a 63.4× runtime-term reduction. DR-12 selects the reference-fitted use of this representation. |
| DR-12 | **selected for production** | Ship the accepted mono `current_radii` grouped carrier on the four fixed Metal hits. Do not widen the mono engine contract for the stereo evidence asset. | User listening accepted the mono fitted impulse. The shipping version must be refitted and re-approved natively at 44.1 kHz before integration; architecture, group count, and output contract are frozen. |
| DR-13 | **selected for production** | Add bipolar `POSITION` over the classic anti-causal grouped room and accepted forward room; do not reuse the incumbent whole-composite `dir` mirror. Prefer the direct dual evaluator and use a fixed-scene mono basis only if measured reserve fails. | Analytic reverse matches literal reverse/source/room/reverse at `1.88e-10`/`2.55e-10` NRMSE. Incumbent mirror NRMSE is `1.0`. Direct midpoint cost is 1,152 four-hit equivalents; native mono cached fallback is 5.38 MiB. |
| DR-14 | **frozen production boundary** | Add a dedicated terminal `groupedroom` modal-to-signal node and versioned immutable binary asset binding; route only the four Metal hits, remove chord-room sends, replace LENGTH with a 20 ms glided POSITION, and defer size/decay variants. | Prevents grouped carriers from being flattened or assigned incumbent direction semantics. Avoids JSON literal expansion and per-tile static-data upload. Full contract: `06-room-position-production-handoff.md`. |
| DR-15 | **frozen integration base** | Land room production directly atop `15e5a39` on `demo/modal-pocket-scene`, followed by the commit containing the handoff; do not restart from `91ecf0c`, `main`, or a historical room lane. | The checkpoint already integrates and validates the scope/control, epoch, morph, telemetry, capture, and qualification prerequisites. Recreating the room on the baseline would silently omit them. |

## Remaining major decisions

1. The native 44.1 kHz regeneration must retain the accepted room by listening;
   failure blocks integration without silently changing architecture.
2. Release-Mac timing selects direct analytic evaluation or its documented
   5.38 MiB mono fixed-scene fallback.
3. If the exact reverse endpoint is not unmistakable in the integrated scene,
   stop for user review before auditioning a longer decay or transient-only
   room send.
4. Final headphone/monitor acceptance of the integrated listening capture.

Room size, stereo output, and a longer default tail are deferred, not active
production questions. All other choices follow the handoff, master fallback
ladder, and stop-the-line rules.
