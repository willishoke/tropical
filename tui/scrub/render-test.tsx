// Headless UI test: mount the real App (engine spawns, patch loads, audio
// OFF), drive some input, assert it renders + survives. Run:
//   TUI_NO_AUDIO=1 bun tui/render-test.tsx
import React from 'react'
import { render } from 'ink-testing-library'
import { App } from './app'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

const main = async () => {
  process.env.TUI_NO_AUDIO = '1'
  const { lastFrame, stdin, unmount } = render(<App />)
  await sleep(1500) // let engine boot + patch load

  let frame = lastFrame() ?? ''
  if (!frame.includes('reversible')) throw new Error('missing title:\n' + frame)
  if (!frame.includes('loaded (audio off)')) throw new Error('patch did not load:\n' + frame)
  if (!frame.includes('Velocity')) throw new Error('params not rendered:\n' + frame)

  // velocity is row 0 — scrub it negative (reverse) and check transport label
  for (let i = 0; i < 30; i++) stdin.write('\x1B[D') // left arrow ×30
  await sleep(300)
  frame = lastFrame() ?? ''
  if (!frame.includes('REVERSE')) throw new Error('reverse transport not shown:\n' + frame)

  // toggle reset
  stdin.write('r')
  await sleep(200)

  unmount()
  console.log('UI OK — mounted, loaded patch, rendered params, handled input')
  await sleep(100)
  process.exit(0)
}
main().catch((e) => { console.error('UI FAIL:', e.message); process.exit(1) })
