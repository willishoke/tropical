# Nested
```tropical
program Outer(f: float) -> (out: float) {
  program Inner(g: float) -> (sig: float) {
    sig = g * 2
  }
  i = Inner(g: f)
  out = i.sig
}
```
