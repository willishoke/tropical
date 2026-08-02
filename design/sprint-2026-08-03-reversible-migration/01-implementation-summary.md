# Reversible migration implementation summary

This is the sprint's running record of implementation-level decisions,
deviations, qualification status, and remaining obligations. Frozen product
and architecture decisions remain in `00-master.md`.

## Status

- Sprint branch: `sprint/reversible-migration`
- Integration baseline: `07a6b2517f8d24c8822d67e0337c0fd99d016bd8`
- Committed handoff: `3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec`
- Corrected 15-commit replay head: `6a0e0c3765ed1ce263b8f96e8c096863ddd28710`
- History proof: passing
- Remote recovery: corrected replay read back from `origin/sprint/reversible-migration`

## Small-scale decisions

### S-01 — Keep the sprint in an isolated worktree

The user's primary checkout contains unrelated untracked design material.
All sprint edits, builds, and commits use `/private/tmp/tropical-reversible-migration-corrected`
so those files remain untouched.

### S-02 — Put the committed handoff before the replay

The handoff exists on the remote as commit `3bf81bc`, directly above the
integration baseline. An initial replay began at the integration baseline,
which preserved all 15 source commits but omitted C0 from the sprint branch.
Before modernization, the branch was rebuilt from the handoff commit, the
full proof was rerun, and the remote ref was corrected with a guarded lease.
This gives the intended order `baseline → handoff → replay`.

### S-03 — Make the import verifier restoration-friendly

The verifier accepts a source-repository argument. It can check either the
untouched standalone checkout or a repository restored from the verified
bundle, so the proof procedure is not coupled to one live checkout path.

## Qualification notes

No release claim has been made. The listening decision, qualified-Mac
hardware gates, ten-minute workload, and lifecycle/package evidence remain
required even if implementation and automated tests are complete.
