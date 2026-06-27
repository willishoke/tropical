// Load the full patch, confirm it compiles, params register, set_param works.
// Does NOT start audio. Run: bun tui/smoke.ts
import { EngineClient, repoRoot } from './client'
import { buildProgram, PARAMS } from './patch'

const c = new EngineClient(repoRoot)
const main = async () => {
  await c.call('list_params') // warm up
  const loaded = await c.call('load', { program: buildProgram() })
  console.log('loaded:', JSON.stringify(loaded))
  const params = await c.call('list_params')
  const names = (params as any[]).map((p) => p.name).sort()
  console.log(`params registered (${names.length}):`, names.join(', '))
  const missing = PARAMS.filter((p) => !names.includes(p.name)).map((p) => p.name)
  if (missing.length) throw new Error('MISSING params: ' + missing.join(', '))
  // drive a few to confirm slots are live — including reverse + freeze
  await c.call('set_param', { name: 'velocity', value: -1 }) // reverse
  await c.call('set_param', { name: 'velocity', value: 0 })  // freeze
  await c.call('set_param', { name: 'f0', value: 165 })
  await c.call('set_param', { name: 'master_gain', value: 0.5 })
  console.log('set_param OK — τ transport live (forward / reverse / freeze)')
  c.kill()
}
main().catch((e) => { console.error('FAIL:', e.message); c.kill(); process.exit(1) })
