# MatchTag
```tropical
program MT(sel: float) -> (out: float) {
  enum Mode { A, B(gain: float) }
  reg m: Mode = A
  next m = B { gain: sel }
  out = match m { A => 0, B { gain: g } => g }
}
```
