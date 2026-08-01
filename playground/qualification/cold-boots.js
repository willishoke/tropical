#!/usr/bin/env node
'use strict'

const { spawnSync } = require('node:child_process')
const { mkdirSync, readFileSync, writeFileSync } = require('node:fs')
const { resolve, join } = require('node:path')

const REPO = resolve(__dirname, '../..')

function parseArgs(argv) {
  const options = {
    count: 5,
    durationSeconds: 5,
    output: join(REPO, 'benchmarks/demo_release/data'),
  }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    const next = () => argv[++index]
    if (arg === '--count') options.count = Number(next())
    else if (arg === '--duration-seconds') options.durationSeconds = Number(next())
    else if (arg === '--output') options.output = resolve(next())
    else if (arg === '--help') options.help = true
    else throw new Error(`unknown argument: ${arg}`)
  }
  return options
}

function usage() {
  return `usage: node playground/qualification/cold-boots.js [options]

  --count N                       independent engine/DAC boots (default 5)
  --duration-seconds N            measured seconds per boot (default 5)
  --output DIR                    evidence directory`
}

function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    process.stdout.write(`${usage()}\n`)
    return
  }
  if (!Number.isInteger(options.count) || options.count < 1) {
    throw new Error('count must be a positive integer')
  }
  if (!Number.isFinite(options.durationSeconds)
      || options.durationSeconds <= 0) {
    throw new Error('duration must be positive')
  }
  mkdirSync(options.output, { recursive: true })
  const started = new Date().toISOString()
  const runs = []
  for (let index = 0; index < options.count; index += 1) {
    const child = spawnSync(process.execPath, [
      join(__dirname, 'run.js'),
      '--mode', 'smoke',
      '--duration-seconds', String(options.durationSeconds),
      '--output', options.output,
    ], {
      cwd: REPO,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    const summaryPath = child.stdout.trim().split('\n').filter(Boolean).at(-1)
    let summary = null
    if (summaryPath) {
      try { summary = JSON.parse(readFileSync(summaryPath, 'utf8')) } catch {}
    }
    runs.push({
      boot: index + 1,
      exit_code: child.status,
      summary_path: summaryPath || null,
      run_id: summary?.run_id ?? null,
      pass: child.status === 0 && summary?.pass === true,
      error: child.stderr.trim() || null,
    })
    if (!runs.at(-1).pass) break
  }
  const aggregate = {
    schema: 1,
    kind: 'cold_boot_matrix',
    started,
    finished: new Date().toISOString(),
    requested_boots: options.count,
    completed_boots: runs.length,
    duration_seconds_each: options.durationSeconds,
    pass: runs.length === options.count && runs.every((run) => run.pass),
    runs,
  }
  const id = started.replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
  const outputPath = join(options.output, `${id}-cold-boots.summary.json`)
  writeFileSync(outputPath, `${JSON.stringify(aggregate, null, 2)}\n`)
  process.stdout.write(`${outputPath}\n`)
  if (!aggregate.pass) process.exitCode = 1
}

try { main() }
catch (error) {
  process.stderr.write(`${error.stack || error}\n`)
  process.exitCode = 1
}
