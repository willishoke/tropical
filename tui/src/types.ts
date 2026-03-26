/** Discriminated union for log entries displayed in the OutputLog. */
export type LogEntry =
  | { kind: "command"; text: string }
  | { kind: "success"; text: string }
  | { kind: "error"; text: string }
  | { kind: "info"; text: string };
