/**
 * compiler/ir/slots.ts — slot-table allocation for resolved programs.
 *
 * Single owner of the decl-identity → slot-integer mapping that the C++
 * engine consumes. Both `loadProgramDefFromResolved` (legacy emit
 * boundary) and `compileResolved` (new emit boundary) call
 * `buildSlotMaps` to allocate slots in the same order.
 *
 * Slot order:
 *   inputs:    in port-declaration order
 *   regs:      in body-decl order  (post-Phase-0a: includes former
 *              delays, since the IR now has one unified state primitive
 *              `RegDecl { name, init, update? }`. A reg with `update`
 *              undefined holds its current value; a reg with `update`
 *              defined evaluates the expression each sample to produce
 *              the next value. The slot allocation orders all of them
 *              by body position — this differs from the pre-0a
 *              "regs-then-delays" segregation only when source code
 *              interleaves reg/delay declarations.)
 *   instances: in body-decl order  (pre-`inlineInstances` only)
 *
 * No name parsing: the maps key on decl object identity.
 */

import type {
  ResolvedProgram, RegDecl, InstanceDecl, InputDecl,
} from './nodes.js'

export interface Slots {
  inputs:    Map<InputDecl, number>
  regs:      Map<RegDecl, number>
  instances: Map<InstanceDecl, number>
}

/** Allocate slot indices for every decl in `prog`. Returns the slot
 *  maps plus the parallel arrays of decls (caller-friendly: indexed
 *  iteration is the common consumption pattern). */
export interface SlotMaps extends Slots {
  inputDecls:    InputDecl[]
  regDecls:      RegDecl[]
  instanceDecls: InstanceDecl[]
}

export function buildSlotMaps(prog: ResolvedProgram): SlotMaps {
  const slots: Slots = {
    inputs:    new Map(),
    regs:      new Map(),
    instances: new Map(),
  }

  const inputDecls = prog.ports.inputs.slice()
  inputDecls.forEach((d, i) => slots.inputs.set(d, i))

  const regDecls:      RegDecl[]      = []
  const instanceDecls: InstanceDecl[] = []

  for (const decl of prog.body.decls) {
    switch (decl.op) {
      case 'regDecl':
        slots.regs.set(decl, regDecls.length)
        regDecls.push(decl)
        break
      case 'instanceDecl':
        slots.instances.set(decl, instanceDecls.length)
        instanceDecls.push(decl)
        break
      case 'paramDecl':   /* session-scoped; not part of slot allocation */ break
      case 'programDecl': /* type-decl only; nothing runtime-shaped */ break
    }
  }

  return { ...slots, inputDecls, regDecls, instanceDecls }
}
