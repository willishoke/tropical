import Tropical.Frontend

/-!
The `frontend` executable: the tropical MCP server. All logic lives in
the `Tropical` library (`Tropical/Frontend.lean` et al.); this file is
just the entry point.
-/

def main : IO Unit := Tropical.runFrontend
