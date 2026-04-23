/**
 * dev.ts — lightweight static server for web/dist with the cross-origin
 * isolation headers SharedArrayBuffer requires.
 *
 *   bun web/dev.ts [--port 8080]
 */

import { resolve, dirname, join, extname } from 'path'
import { readFileSync, existsSync, statSync } from 'fs'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(__dirname, 'dist')

const args = process.argv.slice(2)
const portIdx = args.indexOf('--port')
const port = portIdx >= 0 ? parseInt(args[portIdx + 1] ?? '8080', 10) : 8080

const MIME: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.mjs':  'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.css':  'text/css; charset=utf-8',
}

function headers(ct: string): HeadersInit {
  return {
    'content-type': ct,
    'cross-origin-opener-policy': 'same-origin',
    'cross-origin-embedder-policy': 'require-corp',
    'cache-control': 'no-store',
  }
}

Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url)
    let path = decodeURIComponent(url.pathname)
    if (path === '/') path = '/index.html'
    const full = resolve(distDir, '.' + path)

    if (!full.startsWith(distDir)) return new Response('Forbidden', { status: 403 })
    if (!existsSync(full) || !statSync(full).isFile()) {
      return new Response(`Not found: ${path}`, { status: 404 })
    }
    const ext = extname(full).toLowerCase()
    const ct = MIME[ext] ?? 'application/octet-stream'
    return new Response(readFileSync(full), { headers: headers(ct) })
  },
})

console.log(`▸ serving ${distDir} on http://localhost:${port}`)
console.log(`  COOP/COEP enabled for SharedArrayBuffer`)
