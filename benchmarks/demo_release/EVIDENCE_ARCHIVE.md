# Demo qualification evidence archive

The raw qualification streams and audio captures for the modal-pocket demo are
stored in the immutable GitHub Release
[`demo-modal-pocket-qualification-2026-08-02`](https://github.com/willishoke/tropical/releases/tag/demo-modal-pocket-qualification-2026-08-02).

The archive was created from commit
`07a6b2517f8d24c8822d67e0337c0fd99d016bd8` before repository-tip
compaction. It contains the committed contents of:

- `benchmarks/demo_release/data/`; and
- `benchmarks/demo_release/room_audition/release_out/`.

| Asset | SHA-256 |
|---|---|
| `tropical-modal-pocket-demo-evidence-2026-08-02-07a6b25.tar.gz` | `4a9b0796f46b91df816ee8ce2261c0a5ae2469e1ab261ba402cdc189630cf5e0` |

The release also publishes the checksum as a separate asset. The uploaded
archive was downloaded again, checksum-verified, gzip-tested, and confirmed to
contain 75 entries before raw duplicates were removed from the branch tip.

Compact summaries in `data/` retain aggregate measurements and gates. Their
per-gesture `gesture_delivery` arrays are present in the archive and replaced
in Git by an archive pointer and original entry count. Manifests remain in Git
to preserve candidate, graph, machine, toolchain, device, and worktree truth.

The continuous listening WAVs are archived evidence, not runtime inputs. Their
hashes and aggregate levels remain in
`room_audition/release_out/release_listening_summary.json`. Archival does not
change the recorded status of subjective listening gates.
