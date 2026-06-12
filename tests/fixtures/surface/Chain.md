# Chain
```tropical
program ChainTest(x: float) -> (out: float) {
  out = chain(8, x, (v) => v + 1)
}
```
