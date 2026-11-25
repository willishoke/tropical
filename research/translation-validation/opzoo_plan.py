"""A synthetic tropical_plan_7 that applies every integer op the symbolic
carrier implements to free params (int32 a, b; int c, d), one op per output
channel, so the carrier's dispatch (sorts, coercions, total edges) can be
tested differentially against Semantics.Scalar (diff_ref.py) -- a bounded
check of the open dispatch obligation, not a proof.
usage: opzoo_plan.py <template plan.json> <out plan.json>  (prints the channel names)"""
import json, sys

def S(i, t): return {'index': i, 'kind': 'slot', 'scalar_type': t}
def C(v, t): return {'kind': 'const', 'scalar_type': t, 'val': v}
def R(r, t): return {'kind': 'reg', 'scalar_type': t, 'slot': r}
A, B, Cc, D = S(0, 'int32'), S(1, 'int32'), S(2, 'int'), S(3, 'int')

# (name, tag, result type, args); bool results are wrapped in Select(_, 1, 0)
ops = []
for w, x, y in (('int32', A, B), ('int', Cc, D)):
    for tag in ('Add', 'Sub', 'Mul', 'Div', 'Mod', 'FloorDiv', 'BitAnd', 'BitOr', 'BitXor', 'LShift', 'RShift'):
        ops.append(('%s_%s' % (tag, w), tag, w, [x, y]))
    for k in (0, 1, 15, 31, 33, 63, 64, -1):
        ops.append(('RShiftK%d_%s' % (k, w), 'RShift', w, [x, C(k, w)]))
        ops.append(('LShiftK%d_%s' % (k, w), 'LShift', w, [x, C(k, w)]))
    for tag in ('Neg', 'Abs', 'BitNot'):
        ops.append(('%s_%s' % (tag, w), tag, w, [x]))
    ops.append(('Clamp_%s' % w, 'Clamp', w, [x, C(-1000, w), y]))
    for tag in ('Less', 'LessEq', 'Greater', 'GreaterEq', 'Equal', 'NotEqual'):
        ops.append(('%s_%s' % (tag, w), tag, 'bool', [x, y]))
    ops.append(('Not_%s' % w, 'Not', 'bool', [x]))
    ops.append(('ToBool_%s' % w, 'ToBool', 'bool', [x]))
    ops.append(('DivMin_%s' % w, 'Div', w, [C(-2**31 if w == 'int32' else -2**63, w), y]))
ops += [('ToInt_from32', 'ToInt', 'int', [A]), ('ToInt32_from64', 'ToInt32', 'int32', [Cc]),
        ('SelectSym', 'Select', 'int32', [A, B, C(7, 'int32')])]

tmpl = json.load(open(sys.argv[1]))
nparam = 4
slot_types = [{'slot': 0, 'scalar': 'int32'}, {'slot': 1, 'scalar': 'int32'},
              {'slot': 2, 'scalar': 'int'}, {'slot': 3, 'scalar': 'int'}]
instrs, reg = [], 0
def emit(tag, rt, args):
    global reg
    instrs.append({'args': args, 'dst': reg, 'dst_kind': 'temp', 'loop_count': 1,
                   'result_type': rt, 'strides': [], 'tag': tag})
    reg += 1
    return reg - 1
# every channel is int32 (one output element type): an int result is read out
# as its low and high words
chan = []
for name, tag, rt, args in ops:
    r = emit(tag, rt, args)
    if rt == 'bool':
        r = emit('Select', 'int32', [R(r, 'bool'), C(1, 'int32'), C(0, 'int32')]); rt = 'int32'
    if rt == 'int':
        lo = emit('ToInt32', 'int32', [R(r, 'int')])
        hi = emit('ToInt32', 'int32', [R(emit('RShift', 'int', [R(r, 'int'), C(32, 'int')]), 'int')])
        chan += [(name + '.lo', lo), (name + '.hi', hi)]
    else:
        chan.append((name, r))
slot_names = ['param:a', 'param:b', 'param:c', 'param:d'] + ['out:' + c[0] for c in chan]
for i, (name, r) in enumerate(chan):
    instrs.append({'args': [R(r, 'int32')], 'dst': nparam + i, 'dst_kind': 'moduleSlot', 'loop_count': 1,
                   'result_type': 'int32', 'strides': [], 'tag': 'WriteSlot'})
    slot_types.append({'slot': nparam + i, 'scalar': 'int32'})
plan = dict(tmpl)
child = dict(tmpl['instance_functions'][0]['children'][0])
child.update({'instance_name': 'zoo', 'name': 'instance_zoo', 'instructions': instrs,
              'pre_input_instructions': [], 'register_count': reg, 'register_offset': 0})
root = dict(tmpl['instance_functions'][0]); root.update({'children': [child], 'register_offset': reg})
plan.update({'instance_functions': [root], 'register_count': reg, 'slot_count': len(slot_names),
             'slot_names': slot_names, 'slot_defaults': [0] * len(slot_names), 'slot_types': slot_types,
             'sinks': [{'gain': 1, 'inputs': [nparam + i], 'target': i} for i in range(len(chan))],
             'output_channel_count': len(chan)})
json.dump(plan, open(sys.argv[2], 'w'))
print(' '.join(c[0] for c in chan))
