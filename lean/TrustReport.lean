import Tropical.Trust

def main (args : List String) : IO UInt32 := do
  if args.contains "--sites" then
    IO.print Tropical.Trust.renderTrustSites
    return 0
  let report := Tropical.Trust.renderMarkdown
  if args.contains "--write" then
    IO.FS.writeFile "design/trust-boundary.md" report
    IO.println "wrote design/trust-boundary.md"
    return 0
  if args.contains "--check" then
    let path := System.FilePath.mk "design/trust-boundary.md"
    if !(← path.pathExists) then
      IO.eprintln "design/trust-boundary.md is missing"
      return 1
    if (← IO.FS.readFile path) == report then
      IO.println "trust report is current"
      return 0
    IO.eprintln "design/trust-boundary.md differs from Tropical.Trust.renderMarkdown"
    return 1
  IO.print report
  return 0
