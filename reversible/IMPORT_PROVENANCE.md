# Reversible import provenance

## Frozen source

```text
source_path=/Users/willishoke/reversible
source_head=dea822ea1062a749ea1d7a76af1e2bd28194dfa1
source_root=e7693fa412dfd0082341763a62682ce64a46f8cf
source_commit_count=15
source_branch=main
source_remotes=none
source_tags=none
target_base=07a6b2517f8d24c8822d67e0337c0fd99d016bd8
target_handoff=3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec
target_prefix=reversible/
```

The standalone source was clean when frozen. It remains intact at the source
path and was not rewritten during the import.

## Backup

```text
bundle_path=/Users/willishoke/reversible-before-tropical-2026-08-02.bundle
bundle_sha256=b60ee44f9ac18cb9f43be973bff6cd17d530325cc6b93a965fb4fe071bafd5e1
bundle_size=49958
bundle_verify=complete history; refs/heads/main and HEAD at dea822ea1062a749ea1d7a76af1e2bd28194dfa1
```

The bundle was created with `git bundle create ... --all`, verified with
`git bundle verify`, and hashed with `shasum -a 256` before replay began.

## Replay

The 15 source commits were exported in chronological order using
`git format-patch --root --binary --full-index` and applied to the committed
sprint handoff with `git am --directory=reversible`. The handoff is a direct
child of the frozen integration baseline. No replayed commit was amended,
squashed, skipped, or resolved with `--3way`.

The first attempted remote replay began directly at `target_base`, omitting
the already-committed handoff from the branch. Before any modernization edit,
the branch was rebuilt from `target_handoff`, proven again, and replaced with
`--force-with-lease` while the incorrect replay head was still
`cf22c55c941c27bd2d9c688bbed459ee5e440e36`. The corrected replay head is
`6a0e0c3765ed1ce263b8f96e8c096863ddd28710`.

## Verification and remote recovery

Run the committed verifier from the Tropical worktree:

```bash
reversible/scripts/verify-import.sh /Users/willishoke/reversible
```

The source argument may instead name a repository restored from the verified
bundle. The verifier checks counts, linear parentage, author identity and
timestamp, full commit message, recursive tree object identity, final subtree
identity, and prefix containment for every mapped commit.

The corrected replay was pushed to `origin/sprint/reversible-migration` and
read back at `6a0e0c3765ed1ce263b8f96e8c096863ddd28710` before this post-import
provenance commit was created.
