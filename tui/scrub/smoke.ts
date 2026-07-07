// Load the patch graph, confirm it compiles, the knob slots register, and each
// param drives live over its RPC. Does NOT start audio. Run: bun tui/scrub/smoke.ts
import { EngineClient, repoRoot } from './client'
import { buildGraph, PARAMS, type ParamMode } from './patch'

const METHOD: Record<ParamMode, string> = {
  live: 'set_param',
  glide: 'set_param',
  freq: 'set_param',
  velocity: 'set_param',
}
// The seeded-slot name a mode's drive addresses: a glided knob lives in its
// `#v0` ramp slot; a raw/anchored knob in the bare slot (see Playground.collectParams).
const seededName = (p: { slot: string; mode: ParamMode }) =>
  p.mode === 'glide' ? `${p.slot}#v0` : p.slot

const c = new EngineClient(repoRoot)
const main = async () => {
  await c.call('list_params') // warm up
  const loaded = await c.call('load_patch_graph', buildGraph())
  console.log('loaded:', JSON.stringify(loaded))
  const params = await c.call('list_params')
  const names = (params as any[]).map((p) => p.name).sort()
  console.log(`slots seeded (${names.length}):`, names.join(', '))
  const missing = PARAMS.filter((p) => !names.includes(seededName(p))).map((p) => p.slot)
  if (missing.length) throw new Error('MISSING slots: ' + missing.join(', '))
  // drive every param over its own RPC to confirm the slots are live.
  for (const p of PARAMS) {
    await c.call(METHOD[p.mode], { name: p.slot, value: p.value })
  }
  console.log('drive OK — unified set_param dispatches raw/glide/anchor/velocity engine-side')
  c.kill()
}
main().catch((e) => { console.error('FAIL:', e.message); c.kill(); process.exit(1) })
