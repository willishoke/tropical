# ZipWith
```tropical
program ZipTest(a: float[4], b: float[4]) -> (out: float[4]) {
  out = zipWith(a, b, (x, y) => x + y * 2)
}
```
