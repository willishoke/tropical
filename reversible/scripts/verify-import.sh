#!/bin/sh
set -eu

source_repo=${1:-/Users/willishoke/reversible}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
target_repo=$(git -C "$script_dir" rev-parse --show-toplevel)
map_file="$target_repo/reversible/import-map.tsv"
target_base=07a6b2517f8d24c8822d67e0337c0fd99d016bd8
target_handoff=3bf81bc47ca52ec1e30b72a5a013e34c1241e4ec
expected_source_head=dea822ea1062a749ea1d7a76af1e2bd28194dfa1
expected_source_root=e7693fa412dfd0082341763a62682ce64a46f8cf
expected_count=15

fail() {
  printf 'verify-import: FAIL: %s\n' "$*" >&2
  exit 1
}

test -f "$map_file" || fail "missing import map: $map_file"
git -C "$source_repo" rev-parse --git-dir >/dev/null 2>&1 ||
  fail "source is not a git repository: $source_repo"

source_head=$(git -C "$source_repo" rev-parse HEAD)
source_root=$(git -C "$source_repo" rev-list --max-parents=0 HEAD)
source_count=$(git -C "$source_repo" rev-list --count HEAD)
map_count=$(awk 'NR > 1 && NF == 2 { count++ } END { print count + 0 }' "$map_file")

test "$source_head" = "$expected_source_head" ||
  fail "source head is $source_head, expected $expected_source_head"
test "$source_root" = "$expected_source_root" ||
  fail "source root is $source_root, expected $expected_source_root"
test "$source_count" -eq "$expected_count" ||
  fail "source has $source_count commits, expected $expected_count"
test "$map_count" -eq "$expected_count" ||
  fail "map has $map_count rows, expected $expected_count"
test "$(git -C "$target_repo" rev-parse "$target_handoff^")" = "$target_base" ||
  fail "handoff is not a direct child of the integration baseline"

previous_target=$target_handoff
row=0
while IFS="$(printf '\t')" read -r source_sha target_sha extra; do
  test "$source_sha" = source_sha && continue
  row=$((row + 1))
  test -z "${extra:-}" || fail "row $row has more than two fields"
  git -C "$source_repo" cat-file -e "$source_sha^{commit}" 2>/dev/null ||
    fail "row $row source commit is missing: $source_sha"
  git -C "$target_repo" cat-file -e "$target_sha^{commit}" 2>/dev/null ||
    fail "row $row target commit is missing: $target_sha"

  mapped_source=$(git -C "$source_repo" rev-list --reverse HEAD | sed -n "${row}p")
  test "$source_sha" = "$mapped_source" ||
    fail "row $row source order mismatch"
  actual_parent=$(git -C "$target_repo" rev-parse "$target_sha^")
  test "$actual_parent" = "$previous_target" ||
    fail "row $row target parent is $actual_parent, expected $previous_target"
  parent_count=$(git -C "$target_repo" rev-list --parents -n 1 "$target_sha" | awk '{ print NF - 1 }')
  test "$parent_count" -eq 1 || fail "row $row target is not linear"

  source_meta=$(git -C "$source_repo" show -s \
    --format='%an%x00%ae%x00%at%x00%aI%x00%s%x00%b' "$source_sha" | shasum -a 256 | awk '{ print $1 }')
  target_meta=$(git -C "$target_repo" show -s \
    --format='%an%x00%ae%x00%at%x00%aI%x00%s%x00%b' "$target_sha" | shasum -a 256 | awk '{ print $1 }')
  test "$source_meta" = "$target_meta" || fail "row $row metadata differs"

  source_tree=$(git -C "$source_repo" rev-parse "$source_sha^{tree}")
  target_tree=$(git -C "$target_repo" rev-parse "$target_sha:reversible")
  test "$source_tree" = "$target_tree" || fail "row $row subtree differs"

  outside=$(git -C "$target_repo" diff-tree --no-commit-id --name-only -r \
    "$actual_parent" "$target_sha" | awk '!/^reversible\// { print }')
  test -z "$outside" || fail "row $row changes paths outside reversible/: $outside"
  previous_target=$target_sha
done < "$map_file"

test "$row" -eq "$expected_count" || fail "read $row rows, expected $expected_count"
final_target=$(awk 'END { print $2 }' "$map_file")
test "$(git -C "$target_repo" rev-parse "$final_target:reversible")" = \
  "$(git -C "$source_repo" rev-parse "$expected_source_head^{tree}")" ||
  fail "final imported subtree differs from frozen source head"

printf 'verify-import: PASS (%d commits, final target %s)\n' "$row" "$final_target"
