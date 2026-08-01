'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const { distribution, faultEntries, percentile, summarize } = require('./qualification/metrics.js')
const { assembleCaptureTimeline } = require('./qualification/wav.js')

test('qualification percentiles use the observed upper sample', () => {
  assert.equal(percentile([4, 1, 3, 2], 0.5), 2)
  assert.deepEqual(distribution([4, 1, 3, 2]), {
    count: 4, min: 1, p50: 2, p95: 4, p99: 4, max: 4,
  })
})

test('every sticky runtime, Metal, DAC, and capture fault blocks', () => {
  const telemetry = {
    runtime: { ownership_failure_count: 1 },
    metal: { starvation_count: 2, morph_failure_count: 3 },
  }
  const audio = { stats: { underrunCount: 4, overrunCount: 5 } }
  const names = faultEntries(telemetry, audio, { nonfinite_sample_count: 6 })
    .map((entry) => entry.name)
  assert.deepEqual(names, [
    'ownership_failure_count', 'starvation_count', 'morph_failure_count',
    'underrunCount', 'overrunCount', 'nonfinite_sample_count',
  ])
})

test('capture assembly records gaps without inventing audio', () => {
  const timeline = assembleCaptureTimeline([
    { start_sample_index: 10, samples: [1, 2] },
    { start_sample_index: 13, samples: [3, 4] },
  ])
  assert.deepEqual(timeline.samples, [1, 2, 0, 3, 4])
  assert.equal(timeline.gapSamples, 1)
  assert.equal(timeline.overlapSamples, 0)
})

test('scope metrics exclude intentional preemption and resume frames', () => {
  const summary = summarize({
    writes: [], captures: [], telemetry: {}, audioStatus: {},
    scopes: [
      { wallMs: 0, durationMs: 8 },
      { wallMs: 42, durationMs: 7 },
      { wallMs: 84, durationMs: 30, preempted: true },
      { wallMs: 500, durationMs: 19 },
      { wallMs: 542, durationMs: 6 },
      { wallMs: 584, durationMs: 5, idle: false },
      { wallMs: 626, durationMs: 18 },
    ],
  })
  assert.equal(summary.scope_completed, 6)
  assert.equal(summary.scope_preempted, 1)
  assert.equal(summary.scope_idle_samples, 2)
  assert.deepEqual(summary.scope_rpc_ms, {
    count: 2, min: 6, p50: 6, p95: 7, p99: 7, max: 7,
  })
  assert.equal(summary.scope_frame_interval_ms.count, 2)
  assert.equal(summary.scope_frame_interval_ms.max, 42)
})
