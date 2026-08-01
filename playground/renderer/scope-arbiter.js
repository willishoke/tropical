'use strict';

(function installScopeArbiter(root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.ScopeArbiter = api.ScopeArbiter
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  class ScopeArbiter {
    constructor({ quietMs = 100, now = () => performance.now() } = {}) {
      this.quietMs = quietMs
      this.now = now
      this.active = new Set()
      this.quietUntil = 0
      this.senderBusy = false
      this.faulted = false
    }

    begin(reason) { this.active.add(reason) }

    end(reason) {
      if (!this.active.delete(reason)) return
      this.quietUntil = Math.max(this.quietUntil, this.now() + this.quietMs)
    }

    activity() {
      this.quietUntil = Math.max(this.quietUntil, this.now() + this.quietMs)
    }

    setSenderBusy(busy) {
      if (busy === this.senderBusy) return
      this.senderBusy = busy
      if (!busy) this.activity()
    }

    clearTransient() {
      if (this.active.size > 0 || this.senderBusy) this.activity()
      this.active.clear()
      this.senderBusy = false
    }

    fault() {
      this.clearTransient()
      this.faulted = true
    }

    isHeld() {
      return this.faulted
        || this.senderBusy
        || this.active.size > 0
        || this.now() < this.quietUntil
    }
  }

  return { ScopeArbiter }
}))
