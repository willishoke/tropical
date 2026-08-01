'use strict'

function percentile(values, quantile) {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(quantile * sorted.length) - 1),
  )
  return sorted[index]
}

function distribution(values) {
  return {
    count: values.length,
    min: values.length ? Math.min(...values) : null,
    p50: percentile(values, 0.50),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    max: values.length ? Math.max(...values) : null,
  }
}

const RUNTIME_FAULTS = [
  'ownership_failure_count',
]

const METAL_FAULTS = [
  'dispatch_failure_count',
  'starvation_count',
  'tag_mismatch_count',
  'retarget_count',
  'activation_failure_count',
  'stale_completion_count',
  'callback_thread_violation_count',
  'morph_failure_count',
]

function faultEntries(telemetry, audioStatus, capture = {}) {
  const faults = []
  for (const name of RUNTIME_FAULTS) {
    const value = Number(telemetry?.runtime?.[name] || 0)
    if (value) faults.push({ source: 'runtime', name, value })
  }
  for (const name of METAL_FAULTS) {
    const value = Number(telemetry?.metal?.[name] || 0)
    if (value) faults.push({ source: 'metal', name, value })
  }
  for (const name of ['underrunCount', 'overrunCount']) {
    const value = Number(audioStatus?.stats?.[name] || 0)
    if (value) faults.push({ source: 'dac', name, value })
  }
  for (const [name, value] of Object.entries(capture)) {
    if (Number(value)) faults.push({ source: 'capture', name, value: Number(value) })
  }
  return faults
}

function summarize({ writes, scopes, captures, telemetry, audioStatus }) {
  const dispatched = writes.filter((event) => event.dispatched)
  const audible = dispatched.filter((event) => event.activation?.audible)
  const scopeCompleted = scopes.filter((event) => !event.preempted && !event.error)
  // Idle scope gates apply only inside uninterrupted completed-frame runs.
  // The first completed frame warms/resumes a held canvas; an intentional
  // preemption gap is neither an idle RPC sample nor an idle cadence sample.
  const scopeIdle = []
  const scopeIntervals = []
  for (let index = 1; index < scopes.length; index += 1) {
    const previous = scopes[index - 1]
    const current = scopes[index]
    if (previous.preempted || previous.error
      || current.preempted || current.error
      || previous.idle === false || current.idle === false) continue
    scopeIdle.push(current)
    scopeIntervals.push(current.wallMs - previous.wallMs)
  }
  const captureFaults = {
    nonfinite_sample_count: captures.reduce(
      (sum, block) => sum + block.samples.filter((value) => !Number.isFinite(value)).length,
      0,
    ),
    clamped_sample_count: captures.reduce(
      (sum, block) => sum + block.samples.filter((value) => Math.abs(value) >= 256).length,
      0,
    ),
    all_zero_block_count: captures.filter(
      (block) => block.samples.length && block.samples.every((value) => value === 0),
    ).length,
  }
  return {
    generated_write_count: writes.filter((event) => !event.dispatched).length,
    dispatched_write_count: dispatched.length,
    scheduled_write_latency_ms: distribution(
      dispatched.map((event) => event.dispatchStartMs - event.generatedMs),
    ),
    audible_activation_latency_ms: distribution(
      audible.map((event) => event.audibleMs - event.generatedMs),
    ),
    scope_rpc_ms: distribution(scopeIdle.map((event) => event.durationMs)),
    scope_frame_interval_ms: distribution(scopeIntervals),
    scope_idle_fps: scopeIntervals.length
      ? 1000 / (scopeIntervals.reduce((sum, value) => sum + value, 0)
        / scopeIntervals.length)
      : null,
    scope_idle_samples: scopeIdle.length,
    scope_completed: scopeCompleted.length,
    scope_preempted: scopes.filter((event) => event.preempted).length,
    scope_errors: scopes.filter((event) => event.error).length,
    capture_blocks: captures.length,
    capture_faults: captureFaults,
    faults: faultEntries(telemetry, audioStatus, captureFaults),
  }
}

module.exports = { distribution, faultEntries, percentile, summarize }
