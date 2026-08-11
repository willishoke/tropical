import Tropical.EmitArrow.Modal.OrientedRealize

/-!
# Exact factored terminal for two ordinary rooms

The generic oriented `Bank` algebra remains the reference and fallback.  This
module recognizes the one production shape for which the complete terminal can
be assembled from three shared Cauchy images: a degree-zero source followed by
two degree-zero rooms with the same structural frequency topology.

For `N` source rows and `M` room rows the mapped phases evaluate exactly

* `4*N*M` source/room complex inverses;
* `M*(M-1)/2` symmetry-reduced room-difference inverses; and
* `M*(M+1)/2` symmetry-reduced physical-sum inverses.

Thus the public `N=6`, `M=32` product has 1,792 real reciprocal sites.  Each
complex inverse spells the norm reciprocal once and shares it between the real
and imaginary components.  Direction is absent from all three images and is
applied only by the final `O(N+M)` assembly.
-/

namespace Tropical.EmitArrow.Oriented

open Tropical.Ir
open Tropical.EmitArrow

/-- One checked, complex-valued slice in a routed result image. -/
structure FactoredSlice where
  name : String
  offset : Nat
  extent : Nat
deriving Repr, BEq, Inhabited

private instance : Inhabited ModalMode :=
  ⟨{ sigma := lit 0, omega := lit 0, cre := lit 0 }⟩

def FactoredSlice.floatCount (slice : FactoredSlice) : Nat := 2 * slice.extent

private def FactoredSlice.endOffset (slice : FactoredSlice) : Nat :=
  slice.offset + slice.floatCount

private def FactoredSlice.read (slice : FactoredSlice) (image : Sig)
    (index : Nat) : CplxE :=
  (Sig.index image (lit (Int.ofNat (slice.offset + 2 * index))),
   Sig.index image (lit (Int.ofNat (slice.offset + 2 * index + 1))))

/-- Named source/room image.  The sequential allocator makes overlap
    impossible; `valid` is still checked at the construction seam so routes can
    never silently address beyond the advertised result image. -/
structure SourceRoomLayout where
  d1 : FactoredSlice
  p1 : FactoredSlice
  e1 : FactoredSlice
  e1Cold2 : FactoredSlice
  m1 : FactoredSlice
  d2 : FactoredSlice
  p2 : FactoredSlice
  e2 : FactoredSlice
  m2 : FactoredSlice
  tf : FactoredSlice
  tp : FactoredSlice
  floatCount : Nat
deriving Repr, BEq

private def SourceRoomLayout.create (n m : Nat) : SourceRoomLayout :=
  let d1 := { name := "D[0]", offset := 0, extent := n : FactoredSlice }
  let p1 := { name := "P[0]", offset := d1.endOffset, extent := n : FactoredSlice }
  let d2 := { name := "D[1]", offset := p1.endOffset, extent := n : FactoredSlice }
  let p2 := { name := "P[1]", offset := d2.endOffset, extent := n : FactoredSlice }
  -- Room-indexed slices follow the source-indexed prefix so splitting the map
  -- by room columns can use compact local extents while every source slice
  -- retains one common offset across parts.
  let e1 := { name := "E[0]", offset := p2.endOffset, extent := m : FactoredSlice }
  -- The room-pair diagonal is already present in the exact three-pole
  -- confluence when a source crosses room 2.  Mask those rows before the
  -- reduction instead of subtracting two large summaries after it: the latter
  -- is algebraically exact but catastrophically ill-conditioned in float32.
  let e1Cold2 := { name := "E[0|cold 2]", offset := e1.endOffset, extent := m : FactoredSlice }
  let m1 := { name := "M[0]", offset := e1Cold2.endOffset, extent := m : FactoredSlice }
  let e2 := { name := "E[1]", offset := m1.endOffset, extent := m : FactoredSlice }
  let m2 := { name := "M[1]", offset := e2.endOffset, extent := m : FactoredSlice }
  let tf := { name := "TF", offset := m2.endOffset, extent := m : FactoredSlice }
  let tp := { name := "TP", offset := tf.endOffset, extent := m : FactoredSlice }
  { d1, p1, e1, e1Cold2, m1, d2, p2, e2, m2, tf, tp,
    floatCount := tp.endOffset }

private def SourceRoomLayout.slices (layout : SourceRoomLayout) : Array FactoredSlice :=
  #[layout.d1, layout.p1, layout.d2, layout.p2, layout.e1,
    layout.e1Cold2, layout.m1, layout.e2, layout.m2, layout.tf, layout.tp]

private def SourceRoomLayout.valid (layout : SourceRoomLayout) : Bool :=
  let slices := layout.slices
  slices.zipIdx.all fun (slice, index) =>
    slice.extent > 0 && slice.endOffset <= layout.floatCount &&
      (index == 0 || slices[index - 1]!.endOffset == slice.offset)

structure RoomPairLayout where
  left : FactoredSlice
  right : FactoredSlice
  leftPrime : FactoredSlice
  rightPrime : FactoredSlice
  floatCount : Nat
deriving Repr, BEq

private def RoomPairLayout.create (leftName rightName : String)
    (m : Nat) : RoomPairLayout :=
  let left := { name := leftName, offset := 0, extent := m : FactoredSlice }
  let right := { name := rightName, offset := left.endOffset, extent := m : FactoredSlice }
  let leftPrime := { name := leftName ++ "'", offset := right.endOffset, extent := m : FactoredSlice }
  let rightPrime := { name := rightName ++ "'", offset := leftPrime.endOffset, extent := m : FactoredSlice }
  { left, right, leftPrime, rightPrime, floatCount := rightPrime.endOffset }

private def RoomPairLayout.valid (layout : RoomPairLayout) : Bool :=
  layout.left.extent > 0 && layout.right.extent == layout.left.extent &&
    layout.left.offset == 0 && layout.left.endOffset == layout.right.offset &&
    layout.right.endOffset == layout.leftPrime.offset &&
    layout.leftPrime.endOffset == layout.rightPrime.offset &&
    layout.rightPrime.endOffset == layout.floatCount

/-- Structural cost witness used by qualification tests and plan diagnostics. -/
structure FactoredTerminalStats where
  sourceModes : Nat
  roomModes : Nat
  sourceRoomReciprocals : Nat
  roomDifferenceReciprocals : Nat
  roomPhysicalReciprocals : Nat
  totalReciprocals : Nat
deriving Repr, BEq

def factoredTerminalStats (sourceModes roomModes : Nat) : FactoredTerminalStats :=
  let sourceRoom := 4 * sourceModes * roomModes
  let difference := roomModes * (roomModes - 1) / 2
  let physical := roomModes * (roomModes + 1) / 2
  { sourceModes, roomModes
    sourceRoomReciprocals := sourceRoom
    roomDifferenceReciprocals := difference
    roomPhysicalReciprocals := physical
    totalReciprocals := sourceRoom + difference + physical }

private def cscaleE (scale : Sig) (value : CplxE) : CplxE :=
  (mul scale value.1, mul scale value.2)

/-- One complex inverse with exactly one real reciprocal. -/
private def cinvSharedE (value : CplxE) : CplxE :=
  let invNorm := div (lit 1) (add (mul value.1 value.1) (mul value.2 value.2))
  (mul value.1 invNorm, neg (mul value.2 invNorm))

private def appendComplexRoute (routes : Array (Option Nat))
    (slice : FactoredSlice) (index : Nat) : Array (Option Nat) :=
  routes.push (some (slice.offset + 2 * index))
    |>.push (some (slice.offset + 2 * index + 1))

private def routesInRange (routes : Array (Option Nat)) (outputCount : Nat) : Bool :=
  routes.all fun route => match route with
    | none => true
    | some target => target < outputCount

private def routedImage? (capacity outputCount : Nat)
    (routes : Array (Option Nat)) (tables values : Array Sig)
    (binder : Nat) : Option Sig :=
  if capacity > 0 && outputCount > 0 && !values.isEmpty &&
      routes.size == capacity * values.size && routesInRange routes outputCount
  then some (Sig.routedSum capacity outputCount routes tables values none binder)
  else none

private structure SourceRoomPart where
  image : Sig
  layout : SourceRoomLayout
  roomStart : Nat

private structure SourceRoomImage where
  parts : Array SourceRoomPart

private def SourceRoomImage.images (image : SourceRoomImage) : Array Sig :=
  image.parts.map (·.image)

private def SourceRoomImage.readSource (image : SourceRoomImage)
    (slice : SourceRoomLayout → FactoredSlice) (index : Sig) : CplxE :=
  image.parts.foldl (fun total part =>
    let selected := slice part.layout
    let value : CplxE :=
      (Sig.index part.image
        (add (lit (Int.ofNat selected.offset)) (mul (lit 2) index)),
       Sig.index part.image
        (add (lit (Int.ofNat (selected.offset + 1))) (mul (lit 2) index)))
    caddE total value) (natE 0)

private def SourceRoomImage.readRoom (image : SourceRoomImage)
    (slice : SourceRoomLayout → FactoredSlice) (index : Sig) : CplxE :=
  image.parts.foldl (fun total part =>
    let selected := slice part.layout
    let start := litI (Int.ofNat part.roomStart)
    let stop := litI (Int.ofNat (part.roomStart + selected.extent))
    let owns := Sig.binary .and (Sig.binary .gte index start)
      (Sig.binary .lt index stop)
    let localIndex := sub index start
    let safeLocal := selectE owns localIndex (litI 0)
    let value : CplxE :=
      (Sig.index part.image
        (add (lit (Int.ofNat selected.offset)) (mul (lit 2) safeLocal)),
       Sig.index part.image
        (add (lit (Int.ofNat (selected.offset + 1))) (mul (lit 2) safeLocal)))
    caddE total
      (selectE owns value.1 (lit 0), selectE owns value.2 (lit 0))) (natE 0)

private structure ModeColumns where
  poleRe : Sig
  poleIm : Sig
  ampRe : Sig
  ampIm : Sig

private def modeColumns (modes : Array ModalMode) : ModeColumns where
  poleRe := Sig.arr (modes.map fun mode => mode.poleE.1)
  poleIm := Sig.arr (modes.map fun mode => mode.poleE.2)
  ampRe := Sig.arr (modes.map fun mode => mode.ampE.1)
  ampIm := Sig.arr (modes.map fun mode => mode.ampE.2)

private def ModeColumns.tables (columns : ModeColumns) : Array Sig :=
  #[columns.poleRe, columns.poleIm, columns.ampRe, columns.ampIm]

private def ModeColumns.readPole (columns : ModeColumns) (index : Sig) : CplxE :=
  (Sig.index columns.poleRe index, Sig.index columns.poleIm index)

private def ModeColumns.readAmp (columns : ModeColumns) (index : Sig) : CplxE :=
  (Sig.index columns.ampRe index, Sig.index columns.ampIm index)

private def sourceCrossingHot (delta : CplxE) : Sig :=
  let theta := litF ecddThetaAcc
  Sig.binary .lt (add (mul delta.1 delta.1) (mul delta.2 delta.2))
    (mul theta theta)

/-- Inner source-crossing lane.  Here a quotient of two nearby Cauchy
    transforms is replaced by its analytic derivative.  The outer lens is
    exact, so its boundary agrees algebraically with the ordinary image. -/
private def sourceCrossingInner (delta : CplxE) : Sig :=
  Sig.binary .lt (add (mul delta.1 delta.1) (mul delta.2 delta.2)) (lit 1 12)

private def safeInverseFor (delta : CplxE) (hot : Sig) : CplxE :=
  cinvSharedE (selectE hot (lit 1) delta.1, selectE hot (lit 0) delta.2)

private def unlessHot (hot : Sig) (value : CplxE) : CplxE :=
  (selectE hot (lit 0) value.1, selectE hot (lit 0) value.2)

private def sourceRoomImage? (source room1 room2 : Array ModalMode) :
    Option SourceRoomImage := do
  let n := source.size
  let m := room1.size
  if n == 0 || m == 0 || room2.size != m then none
  let sourceCols := modeColumns source
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let tables := sourceCols.tables ++ room1Cols.tables ++ room2Cols.tables
  let mut parts := #[]
  -- Split by room columns.  Each part retains all source accumulators but only
  -- one quarter of the room-indexed outputs.  Four compact route tables keep
  -- the added crossing correction below the static scratch budget while
  -- retaining one authored reciprocal evaluation per source/room pair.
  for chunk in [0:4] do
    let start := chunk * m / 4
    let stop := (chunk + 1) * m / 4
    if start < stop then
      let width := stop - start
      let capacity := n * width
      let layout := SourceRoomLayout.create n width
      if !layout.valid then none
      let index := Sig.loopIdx 2
      let sourceIndex := Sig.binary .floorDiv index (litI (Int.ofNat width))
      let roomLocal := Sig.binary .mod index (litI (Int.ofNat width))
      let roomIndex := add roomLocal (litI (Int.ofNat start))
      let lam := sourceCols.readPole sourceIndex
      let amp := sourceCols.readAmp sourceIndex
      let pole1 := room1Cols.readPole roomIndex
      let residue1 := room1Cols.readAmp roomIndex
      let pole2 := room2Cols.readPole roomIndex
      let residue2 := room2Cols.readAmp roomIndex
      let q1 := cnegE pole1
      let q2 := cnegE pole2
      let delta1 := csubE lam pole1
      let delta2 := csubE lam pole2
      let hot1 := sourceCrossingHot delta1
      let hot2 := sourceCrossingHot delta2
      let anyHot := Sig.binary .or hot1 hot2
      let u1 := safeInverseFor delta1 hot1
      let v1 := cinvSharedE (csubE q1 lam)
      let u2 := safeInverseFor delta2 hot2
      let v2 := cinvSharedE (csubE q2 lam)
      let d1 := unlessHot hot1 (cmulE residue1 u1)
      let p1 := cmulE residue1 v1
      let e1 := unlessHot hot1 (cnegE (cmulE amp u1))
      let e1Cold2 := unlessHot hot2 e1
      let m1 := cmulE amp v1
      let d2 := unlessHot hot2 (cmulE residue2 u2)
      let p2 := cmulE residue2 v2
      let e2 := unlessHot hot2 (cnegE (cmulE amp u2))
      let m2 := cmulE amp v2
      let tf := unlessHot anyHot (cmulE amp (cmulE u1 u2))
      let tp := cmulE amp (cmulE v1 v2)
      let values := #[d1.1, d1.2, p1.1, p1.2,
        e1.1, e1.2, e1Cold2.1, e1Cold2.2, m1.1, m1.2,
        d2.1, d2.2, p2.1, p2.2, e2.1, e2.2, m2.1, m2.2,
        tf.1, tf.2, tp.1, tp.2]
      let routes := (Array.range capacity).foldl (fun routes k =>
        let i := k / width
        let j := k % width
        let routes := appendComplexRoute routes layout.d1 i
        let routes := appendComplexRoute routes layout.p1 i
        let routes := appendComplexRoute routes layout.e1 j
        let routes := appendComplexRoute routes layout.e1Cold2 j
        let routes := appendComplexRoute routes layout.m1 j
        let routes := appendComplexRoute routes layout.d2 i
        let routes := appendComplexRoute routes layout.p2 i
        let routes := appendComplexRoute routes layout.e2 j
        let routes := appendComplexRoute routes layout.m2 j
        let routes := appendComplexRoute routes layout.tf j
        appendComplexRoute routes layout.tp j) #[]
      let image ← routedImage? capacity layout.floatCount routes tables values 2
      parts := parts.push { image, layout, roomStart := start }
  let _ ← parts[0]?
  pure { parts }

private structure RoomPairImage where
  images : Array Sig
  layout : RoomPairLayout

private def RoomPairImage.read (image : RoomPairImage)
    (slice : RoomPairLayout → FactoredSlice) (index : Sig) : CplxE :=
  image.images.foldl (fun total part =>
    let selected := slice image.layout
    let value : CplxE :=
      (Sig.index part (add (lit (Int.ofNat selected.offset)) (mul (lit 2) index)),
       Sig.index part (add (lit (Int.ofNat (selected.offset + 1))) (mul (lit 2) index)))
    caddE total value) (natE 0)

private def fourChunks (size : Nat) : Array (Nat × Nat) :=
  (Array.range 4).filterMap fun chunk =>
    let start := chunk * size / 4
    let stop := (chunk + 1) * size / 4
    if start < stop then some (start, stop) else none

private def differencePairs (m : Nat) : Array (Nat × Nat) := Id.run do
  let mut pairs := #[]
  for j in [0:m] do
    for k in [j + 1:m] do pairs := pairs.push (j, k)
  return pairs

private def physicalPairs (m : Nat) : Array (Nat × Nat) := Id.run do
  let mut pairs := #[]
  for j in [0:m] do
    for k in [j:m] do pairs := pairs.push (j, k)
  return pairs

/-- Decode the authored triangular item order without a per-item index column.
    For `withDiagonal=false`, item `t` ranges over `j<k`; for `true`, over
    `j≤k`.  Capacities are small and the discriminants are exact integers well
    below `2^24`, so the float square root followed by floor has an exact integer
    answer on every backend. -/
private def triangularPairIndex (m : Nat) (withDiagonal : Bool)
    (item : Sig) : Sig × Sig :=
  let widthNat := if withDiagonal then 2 * m + 1 else 2 * m - 1
  let width := lit (Int.ofNat widthNat)
  let itemFloat := toFloatE item
  let discriminant := sub (mul width width) (mul (lit 8) itemFloat)
  let jFloat := Sig.unary .floor
    (div (sub width (Sig.unary .sqrt discriminant)) (lit 2))
  let j := toIntE jFloat
  let widthI := litI (Int.ofNat widthNat)
  let rowStart := Sig.binary .floorDiv (mul j (sub widthI j)) (litI 2)
  let offset := sub item rowStart
  let k := add j (add offset (if withDiagonal then litI 0 else litI 1))
  (j, k)

private def differenceImage? (room1 room2 : Array ModalMode) : Option RoomPairImage := do
  let m := room1.size
  if m == 0 || room2.size != m then none
  let layout := RoomPairLayout.create "A" "C" m
  if !layout.valid then none
  let pairs := differencePairs m
  if pairs.isEmpty then
    pure { images := #[Sig.arr (Array.replicate layout.floatCount (lit 0))], layout }
  else
    let room1Cols := modeColumns room1
    let room2Cols := modeColumns room2
    let mut images := #[]
    for (start, stop) in fourChunks pairs.size do
      let index := Sig.loopIdx 3
      let globalIndex := add index (litI (Int.ofNat start))
      let (j, k) := triangularPairIndex m false globalIndex
      let inverse := cinvSharedE
        (csubE (room1Cols.readPole j) (room2Cols.readPole k))
      let inverseConj := (inverse.1, neg inverse.2)
      let square := cmulE inverse inverse
      let squareConj := (square.1, neg square.2)
      let aj := cmulE (room2Cols.readAmp k) inverse
      let ck := cnegE (cmulE (room1Cols.readAmp j) inverse)
      let ak := cmulE (room2Cols.readAmp j) inverseConj
      let cj := cnegE (cmulE (room1Cols.readAmp k) inverseConj)
      let ajPrime := cnegE (cmulE (room2Cols.readAmp k) square)
      let ckPrime := cnegE (cmulE (room1Cols.readAmp j) square)
      let akPrime := cnegE (cmulE (room2Cols.readAmp j) squareConj)
      let cjPrime := cnegE (cmulE (room1Cols.readAmp k) squareConj)
      let values := #[aj.1, aj.2, ck.1, ck.2, ak.1, ak.2, cj.1, cj.2,
        ajPrime.1, ajPrime.2, ckPrime.1, ckPrime.2,
        akPrime.1, akPrime.2, cjPrime.1, cjPrime.2]
      let routes := (pairs.extract start stop).foldl (fun routes (j, k) =>
        appendComplexRoute
          (appendComplexRoute
            (appendComplexRoute
              (appendComplexRoute
                (appendComplexRoute
                  (appendComplexRoute
                    (appendComplexRoute
                      (appendComplexRoute routes layout.left j) layout.right k)
                      layout.left k) layout.right j)
                    layout.leftPrime j) layout.rightPrime k)
                layout.leftPrime k) layout.rightPrime j) #[]
      let tables := room1Cols.tables ++ room2Cols.tables
      let image ← routedImage? (stop - start) layout.floatCount routes tables values 3
      images := images.push image
    pure { images, layout }

private def physicalImage? (room1 room2 : Array ModalMode) : Option RoomPairImage := do
  let m := room1.size
  if m == 0 || room2.size != m then none
  let layout := RoomPairLayout.create "B[0,1]" "B[1,0]" m
  if !layout.valid then none
  let pairs := physicalPairs m
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let tables := room1Cols.tables ++ room2Cols.tables
  let mut images := #[]
  for (start, stop) in fourChunks pairs.size do
    let index := Sig.loopIdx 4
    let globalIndex := add index (litI (Int.ofNat start))
    let (j, k) := triangularPairIndex m true globalIndex
    let inverse := cinvSharedE
      (csubE (cnegE (room2Cols.readPole k)) (room1Cols.readPole j))
    let square := cmulE inverse inverse
    let leftJ := cmulE (room2Cols.readAmp k) inverse
    let rightK := cmulE (room1Cols.readAmp j) inverse
    let leftK := cmulE (room2Cols.readAmp j) inverse
    let rightJ := cmulE (room1Cols.readAmp k) inverse
    let leftJPrime := cmulE (room2Cols.readAmp k) square
    let rightKPrime := cmulE (room1Cols.readAmp j) square
    let leftKPrime := cmulE (room2Cols.readAmp j) square
    let rightJPrime := cmulE (room1Cols.readAmp k) square
    let values := #[leftJ.1, leftJ.2, rightK.1, rightK.2,
      leftK.1, leftK.2, rightJ.1, rightJ.2,
      leftJPrime.1, leftJPrime.2, rightKPrime.1, rightKPrime.2,
      leftKPrime.1, leftKPrime.2, rightJPrime.1, rightJPrime.2]
    let routes := (pairs.extract start stop).foldl (fun routes (j, k) =>
      let routes := appendComplexRoute
        (appendComplexRoute routes layout.left j) layout.right k
      let routes := if j == k then
        routes.push none |>.push none |>.push none |>.push none
      else appendComplexRoute
        (appendComplexRoute routes layout.left k) layout.right j
      let routes := appendComplexRoute
        (appendComplexRoute routes layout.leftPrime j) layout.rightPrime k
      if j == k then
        routes.push none |>.push none |>.push none |>.push none
      else appendComplexRoute
        (appendComplexRoute routes layout.leftPrime k) layout.rightPrime j) #[]
    let image ← routedImage? (stop - start) layout.floatCount routes tables values 4
    images := images.push image
  pure { images, layout }

private structure SourceAssemblyLayout where
  amps : FactoredSlice
  strike : FactoredSlice
  floatCount : Nat

private def SourceAssemblyLayout.create (n : Nat) : SourceAssemblyLayout :=
  let amps := { name := "source-future", offset := 0, extent := n : FactoredSlice }
  let strike := { name := "source-strike", offset := amps.endOffset, extent := 1 : FactoredSlice }
  { amps, strike, floatCount := strike.endOffset }

private structure RoomAssemblyLayout where
  future1 : FactoredSlice
  future2 : FactoredSlice
  past1 : FactoredSlice
  past2 : FactoredSlice
  futureDD : FactoredSlice
  pastDD : FactoredSlice
  strike : FactoredSlice
  floatCount : Nat

private def RoomAssemblyLayout.create (m : Nat) : RoomAssemblyLayout :=
  let future1 := { name := "room-1-future", offset := 0, extent := m : FactoredSlice }
  let future2 := { name := "room-2-future", offset := future1.endOffset, extent := m : FactoredSlice }
  let past1 := { name := "room-1-past", offset := future2.endOffset, extent := m : FactoredSlice }
  let past2 := { name := "room-2-past", offset := past1.endOffset, extent := m : FactoredSlice }
  let futureDD := { name := "room-future-dd", offset := past2.endOffset, extent := m : FactoredSlice }
  let pastDD := { name := "room-past-dd", offset := futureDD.endOffset, extent := m : FactoredSlice }
  let strike := { name := "room-strike", offset := pastDD.endOffset, extent := 1 : FactoredSlice }
  { future1, future2, past1, past2, futureDD, pastDD, strike,
    floatCount := strike.endOffset }

private structure ConfluenceLayout where
  pair1DD : FactoredSlice
  pair1Simple : FactoredSlice
  pair2DD : FactoredSlice
  pair2Simple : FactoredSlice
  triple : FactoredSlice
  pole1 : FactoredSlice
  pole2 : FactoredSlice
  hot1Offset : Nat
  hot2Offset : Nat
  sourceCount : Nat
  floatCount : Nat

private def ConfluenceLayout.create (n : Nat) : ConfluenceLayout :=
  let pair1DD := { name := "source-room-1-dd", offset := 0, extent := n : FactoredSlice }
  let pair1Simple := {
    name := "source-room-1-simple", offset := pair1DD.endOffset,
    extent := n : FactoredSlice }
  let pair2DD := {
    name := "source-room-2-dd", offset := pair1Simple.endOffset,
    extent := n : FactoredSlice }
  let pair2Simple := {
    name := "source-room-2-simple", offset := pair2DD.endOffset,
    extent := n : FactoredSlice }
  let triple := {
    name := "source-room-triple", offset := pair2Simple.endOffset,
    extent := n : FactoredSlice }
  let pole1 := {
    name := "source-room-pole-1", offset := triple.endOffset,
    extent := n : FactoredSlice }
  let pole2 := {
    name := "source-room-pole-2", offset := pole1.endOffset,
    extent := n : FactoredSlice }
  let hot1Offset := pole2.endOffset
  let hot2Offset := hot1Offset + n
  { pair1DD, pair1Simple, pair2DD, pair2Simple, triple, pole1, pole2,
    hot1Offset, hot2Offset, sourceCount := n, floatCount := hot2Offset + n }

private def ConfluenceLayout.valid (layout : ConfluenceLayout) : Bool :=
  layout.sourceCount > 0 && layout.pair1DD.offset == 0 &&
    layout.pair1DD.endOffset == layout.pair1Simple.offset &&
    layout.pair1Simple.endOffset == layout.pair2DD.offset &&
    layout.pair2DD.endOffset == layout.pair2Simple.offset &&
    layout.pair2Simple.endOffset == layout.triple.offset &&
    layout.triple.endOffset == layout.pole1.offset &&
    layout.pole1.endOffset == layout.pole2.offset &&
    layout.pole2.endOffset == layout.hot1Offset &&
    layout.hot2Offset == layout.hot1Offset + layout.sourceCount &&
    layout.floatCount == layout.hot2Offset + layout.sourceCount

private structure ConfluenceAssembly where
  images : Array Sig
  layout : ConfluenceLayout

private structure AssemblyImage (Layout : Type) where
  image : Sig
  layout : Layout

private def sourceAssemblyImage? (source : Array ModalMode)
    (sourceRoom : SourceRoomImage) (direction1 direction2 : Sig) :
    Option (AssemblyImage SourceAssemblyLayout) := do
  let n := source.size
  if n == 0 then none
  let layout := SourceAssemblyLayout.create n
  let columns := modeColumns source
  let index := Sig.loopIdx 5
  let gain1 := caddE
    (cscaleE (sub (lit 1) direction1)
      (sourceRoom.readSource (fun layout => layout.d1) index))
    (cscaleE direction1 (sourceRoom.readSource (fun layout => layout.p1) index))
  let gain2 := caddE
    (cscaleE (sub (lit 1) direction2)
      (sourceRoom.readSource (fun layout => layout.d2) index))
    (cscaleE direction2 (sourceRoom.readSource (fun layout => layout.p2) index))
  let amp := cmulE (columns.readAmp index) (cmulE gain1 gain2)
  let values := #[amp.1, amp.2, amp.1, amp.2]
  let routes := (Array.range n).foldl (fun routes i =>
    appendComplexRoute (appendComplexRoute routes layout.amps i) layout.strike 0) #[]
  let image ← routedImage? n layout.floatCount routes
    (sourceRoom.images ++ columns.tables) values 5
  pure { image, layout }

private def roomAssemblyImage? (room1 room2 : Array ModalMode)
    (sourceRoom : SourceRoomImage) (difference physical : RoomPairImage)
    (direction1 direction2 : Sig) : Option (AssemblyImage RoomAssemblyLayout) := do
  let m := room1.size
  if m == 0 || room2.size != m then none
  let layout := RoomAssemblyLayout.create m
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let index := Sig.loopIdx 6
  let f1 := sub (lit 1) direction1
  let f2 := sub (lit 1) direction2
  let d1 := direction1
  let d2 := direction2
  let b1 := room1Cols.readAmp index
  let b2 := room2Cols.readAmp index
  let otherFuture1 := caddE
    (cscaleE f2 (difference.read (fun layout => layout.left) index))
    (cscaleE d2 (physical.read (fun layout => layout.left) index))
  let future1 := cscaleE f1 (cmulE b1
    (cmulE (sourceRoom.readRoom (fun layout => layout.e1) index) otherFuture1))
  let otherFuture2 := caddE
    (cscaleE f1 (difference.read (fun layout => layout.right) index))
    (cscaleE d1 (physical.read (fun layout => layout.right) index))
  let future2Ordinary := cscaleE f2 (cmulE b2
    (cmulE (sourceRoom.readRoom (fun layout => layout.e2) index) otherFuture2))
  let futureDiagonal := cscaleE (neg (mul f1 f2))
    (cmulE (cmulE b1 b2) (sourceRoom.readRoom (fun layout => layout.tf) index))
  let future2 := caddE future2Ordinary futureDiagonal
  let otherPast1 := caddE
    (cscaleE d2 (difference.read (fun layout => layout.left) index))
    (cscaleE f2 (physical.read (fun layout => layout.left) index))
  let past1 := cscaleE d1 (cmulE b1
    (cmulE (sourceRoom.readRoom (fun layout => layout.m1) index) otherPast1))
  let otherPast2 := caddE
    (cscaleE d1 (difference.read (fun layout => layout.right) index))
    (cscaleE f1 (physical.read (fun layout => layout.right) index))
  let past2Ordinary := cscaleE d2 (cmulE b2
    (cmulE (sourceRoom.readRoom (fun layout => layout.m2) index) otherPast2))
  let pastDiagonal := cscaleE (mul d1 d2)
    (cmulE (cmulE b1 b2) (sourceRoom.readRoom (fun layout => layout.tp) index))
  let past2 := caddE past2Ordinary pastDiagonal
  let futureDDSource := sourceRoom.readRoom (fun layout => layout.e1Cold2) index
  let futureDD := cscaleE (mul f1 f2)
    (cmulE (cmulE b1 b2) futureDDSource)
  let pastDD := cscaleE (mul d1 d2)
    (cmulE (cmulE b1 b2) (sourceRoom.readRoom (fun layout => layout.m1) index))
  let strike := caddE future1 future2
  let values := #[future1.1, future1.2, future2.1, future2.2,
    past1.1, past1.2, past2.1, past2.2,
    futureDD.1, futureDD.2, pastDD.1, pastDD.2, strike.1, strike.2]
  let routes := (Array.range m).foldl (fun routes j =>
    appendComplexRoute
      (appendComplexRoute
        (appendComplexRoute
          (appendComplexRoute
            (appendComplexRoute
              (appendComplexRoute
                (appendComplexRoute routes layout.future1 j) layout.future2 j)
                layout.past1 j) layout.past2 j)
            layout.futureDD j) layout.pastDD j) layout.strike 0) #[]
  let tables := sourceRoom.images ++ difference.images ++ physical.images ++
    room1Cols.tables ++ room2Cols.tables
  let image ← routedImage? m layout.floatCount routes tables values 6
  pure { image, layout }

/-- Fold all `N*M` crossing classifications into one coefficient row per
    source.  Room-frequency separation proves that at most one authored room
    index can contribute to a source row, so poles and hot flags may be routed
    with ordinary sums.  This phase contains only algebra; the expensive
    exponential/DD carriers subsequently run at capacity `N`, not `N*M`. -/
private def confluenceAssemblyImage? (source room1 room2 : Array ModalMode)
    (sourceRoom : SourceRoomImage) (difference physical : RoomPairImage)
    (direction1 direction2 : Sig) : Option ConfluenceAssembly := do
  let n := source.size
  let m := room1.size
  if n == 0 || m == 0 || room2.size != m then none
  let layout := ConfluenceLayout.create n
  if !layout.valid then none
  let sourceCols := modeColumns source
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let capacity := n * m
  let f1 := sub (lit 1) direction1
  let f2 := sub (lit 1) direction2
  let d1 := direction1
  let d2 := direction2
  let mut images := #[]
  for chunk in [0:2] do
    let start := chunk * capacity / 2
    let stop := (chunk + 1) * capacity / 2
    if start < stop then
      let flatIndex := add (Sig.loopIdx 14) (litI (Int.ofNat start))
      let sourceIndex := Sig.binary .floorDiv flatIndex (litI (Int.ofNat m))
      let roomIndex := Sig.binary .mod flatIndex (litI (Int.ofNat m))
      let lam := sourceCols.readPole sourceIndex
      let a := sourceCols.readAmp sourceIndex
      let pole1 := room1Cols.readPole roomIndex
      let b1 := room1Cols.readAmp roomIndex
      let pole2 := room2Cols.readPole roomIndex
      let b2 := room2Cols.readAmp roomIndex
      let delta1 := csubE lam pole1
      let delta2 := csubE lam pole2
      let hot1 := sourceCrossingHot delta1
      let hot2 := sourceCrossingHot delta2
      let inner1 := sourceCrossingInner delta1
      let inner2 := sourceCrossingInner delta2
      let anyHot := Sig.binary .or hot1 hot2
      let sourceD1 := sourceRoom.readSource (fun layout => layout.d1) sourceIndex
      let sourceP1 := sourceRoom.readSource (fun layout => layout.p1) sourceIndex
      let sourceD2 := sourceRoom.readSource (fun layout => layout.d2) sourceIndex
      let sourceP2 := sourceRoom.readSource (fun layout => layout.p2) sourceIndex
      -- These two hot-row exclusions used to occupy four source-indexed
      -- columns in every source/room chunk.  Room-frequency separation proves
      -- there is at most one contributing row, so recomputing them in this
      -- already-mapped confluence phase is exact and reclaims the persistent
      -- image columns for product effects.
      let u1 := safeInverseFor delta1 hot1
      let u2 := safeInverseFor delta2 hot2
      let x12 := unlessHot (Sig.binary .or (Sig.unary .not hot1) hot2)
        (cmulE b2 u2)
      let x21 := unlessHot (Sig.binary .or (Sig.unary .not hot2) hot1)
        (cmulE b1 u1)
      let diffLeft := difference.read (fun layout => layout.left) roomIndex
      let diffRight := difference.read (fun layout => layout.right) roomIndex
      let diffLeftPrime := difference.read
        (fun layout => layout.leftPrime) roomIndex
      let diffRightPrime := difference.read
        (fun layout => layout.rightPrime) roomIndex
      let physLeft := physical.read (fun layout => layout.left) roomIndex
      let physRight := physical.read (fun layout => layout.right) roomIndex
      let physLeftPrime := physical.read
        (fun layout => layout.leftPrime) roomIndex
      let physRightPrime := physical.read
        (fun layout => layout.rightPrime) roomIndex
      let k2Lam := caddE
        (cscaleE f2 (csubE sourceD2 x12)) (cscaleE d2 sourceP2)
      let k2Pole := caddE
        (cscaleE f2 diffLeft) (cscaleE d2 physLeft)
      let k2Prime := caddE
        (cscaleE f2 diffLeftPrime) (cscaleE d2 physLeftPrime)
      let safeDelta1 : CplxE :=
        (selectE inner1 (lit 1) delta1.1,
         selectE inner1 (lit 0) delta1.2)
      let k2QuotientDirect := cdivE (csubE k2Lam k2Pole) safeDelta1
      let k2Quotient : CplxE :=
        (selectE inner1 k2Prime.1 k2QuotientDirect.1,
         selectE inner1 k2Prime.2 k2QuotientDirect.2)
      let pair1Scale := cscaleE f1 (cmulE a b1)
      let pair1DD := unlessHot (Sig.unary .not hot1)
        (cmulE pair1Scale k2Pole)
      let pair1Simple := unlessHot (Sig.unary .not hot1)
        (cmulE pair1Scale k2Quotient)
      let k1Lam := caddE
        (cscaleE f1 (csubE sourceD1 x21)) (cscaleE d1 sourceP1)
      let k1Pole := caddE
        (cscaleE f1 diffRight) (cscaleE d1 physRight)
      let k1Prime := caddE
        (cscaleE f1 diffRightPrime) (cscaleE d1 physRightPrime)
      let safeDelta2 : CplxE :=
        (selectE inner2 (lit 1) delta2.1,
         selectE inner2 (lit 0) delta2.2)
      let k1QuotientDirect := cdivE (csubE k1Lam k1Pole) safeDelta2
      let k1Quotient : CplxE :=
        (selectE inner2 k1Prime.1 k1QuotientDirect.1,
         selectE inner2 k1Prime.2 k1QuotientDirect.2)
      let pair2Scale := cscaleE f2 (cmulE a b2)
      let pair2DD := unlessHot (Sig.unary .not hot2)
        (cmulE pair2Scale k1Pole)
      let pair2Simple := unlessHot (Sig.unary .not hot2)
        (cmulE pair2Scale k1Quotient)
      let triple := unlessHot (Sig.unary .not anyHot)
        (cscaleE (mul f1 f2) (cmulE a (cmulE b1 b2)))
      let selectedPole1 := unlessHot (Sig.unary .not anyHot) pole1
      let selectedPole2 := unlessHot (Sig.unary .not anyHot) pole2
      let values := #[pair1DD.1, pair1DD.2, pair1Simple.1, pair1Simple.2,
        pair2DD.1, pair2DD.2, pair2Simple.1, pair2Simple.2,
        triple.1, triple.2, selectedPole1.1, selectedPole1.2,
        selectedPole2.1, selectedPole2.2, toFloatE hot1, toFloatE hot2]
      let routes := (Array.range capacity |>.extract start stop).foldl
        (fun routes k =>
          let i := k / m
          let routes := appendComplexRoute routes layout.pair1DD i
          let routes := appendComplexRoute routes layout.pair1Simple i
          let routes := appendComplexRoute routes layout.pair2DD i
          let routes := appendComplexRoute routes layout.pair2Simple i
          let routes := appendComplexRoute routes layout.triple i
          let routes := appendComplexRoute routes layout.pole1 i
          let routes := appendComplexRoute routes layout.pole2 i
          routes.push (some (layout.hot1Offset + i))
            |>.push (some (layout.hot2Offset + i))) #[]
      let tables := sourceCols.tables ++ room1Cols.tables ++ room2Cols.tables ++
        sourceRoom.images ++ difference.images ++ physical.images
      let image ← routedImage? (stop - start) layout.floatCount routes
        tables values 14
      images := images.push image
  pure { images, layout }

private def sameFrequencyTopology (room1 room2 : Array ModalMode) : Bool :=
  room1.size == room2.size && (room1.zip room2).all fun (left, right) =>
    left.omega == right.omega

private def degreeZero (modes : Array ModalMode) : Bool :=
  modes.all fun mode => mode.deg == 0

/-- Distinct room rows must have disjoint source-crossing lenses.  Then a
    source row can be hot against at most one authored room index, making the
    `X[1→2]`/`X[2→1]` per-source summaries exact.  Non-constant frequencies
    are conservatively left on the generic terminal until their declared
    intervals can prove the same separation. -/
private def separatedRoomFrequencies (modes : Array ModalMode) : Bool :=
  modes.zipIdx.all fun (left, j) =>
    modes.zipIdx.all fun (right, k) =>
      if j >= k then true else
      match sigConstF? left.omega, sigConstF? right.omega with
      | some l, some r => (l - r).abs >= 2.0 * ecddThetaAcc
      | _, _ => false

/-- A constant imaginary-frequency gap is enough to prove that the full
    complex pole distance never enters the source/room confluence lens.  This
    is especially useful for the real phaser tails (`omega = 0`) against an
    authored room whose modes are all audible-frequency conjugate pairs. -/
private def sourceRoomConfluenceCold (source room1 room2 : Array ModalMode) : Bool :=
  source.all fun sourceMode =>
    (room1 ++ room2).all fun roomMode =>
      match sigConstF? sourceMode.omega, sigConstF? roomMode.omega with
      | some sourceOmega, some roomOmega =>
          (sourceOmega - roomOmega).abs >= ecddThetaAcc
      | _, _ => false

/-- Does this terminal have the exact topology served by the factored image?
    Numeric pole values never influence this decision. -/
def factoredTwoRoomAdmitted (source room1 room2 : Array ModalMode) : Bool :=
  !source.isEmpty && !room1.isEmpty && degreeZero source && degreeZero room1 &&
    degreeZero room2 && sameFrequencyTopology room1 room2 &&
    separatedRoomFrequencies room1

/-- The factored terminal retains the ordinary carrier spelling as data, but
    realizes its four banks through routed float images.  Keeping this wrapper
    distinct prevents the generic `TerminalBank` fallback from accidentally
    acquiring cooperative execution or a different numeric policy. -/
structure FactoredTwoRoomTerminal where
  source : Array ModalMode
  room1 : Array ModalMode
  room2 : Array ModalMode
  direction1 : Sig
  direction2 : Sig
  confluenceImages : Array Sig
  confluencePair1DD : FactoredSlice
  confluencePair1Simple : FactoredSlice
  confluencePair2DD : FactoredSlice
  confluencePair2Simple : FactoredSlice
  confluenceTriple : FactoredSlice
  confluencePole1 : FactoredSlice
  confluencePole2 : FactoredSlice
  confluenceHot1Offset : Nat
  confluenceHot2Offset : Nat
  sourceAssembly : Sig
  sourceAmps : FactoredSlice
  roomAssembly : Sig
  roomFuture1 : FactoredSlice
  roomFuture2 : FactoredSlice
  roomPast1 : FactoredSlice
  roomPast2 : FactoredSlice
  roomFutureDD : FactoredSlice
  roomPastDD : FactoredSlice
  atZero : CplxE

private def sumRoutes (count : Nat) : Array (Option Nat) :=
  Array.replicate count (some 0)

private def FactoredSlice.readDynamic (slice : FactoredSlice) (image index : Sig) : CplxE :=
  (Sig.index image (add (lit (Int.ofNat slice.offset)) (mul (lit 2) index)),
   Sig.index image (add (lit (Int.ofNat (slice.offset + 1))) (mul (lit 2) index)))

private def readImageParts (images : Array Sig) (slice : FactoredSlice)
    (index : Sig) : CplxE :=
  images.foldl (fun total image =>
    caddE total (slice.readDynamic image index)) (natE 0)

private def readScalarParts (images : Array Sig) (offset : Nat)
    (index : Sig) : Sig :=
  images.foldl (fun total image =>
    add total (Sig.index image (add (litI (Int.ofNat offset)) index))) (lit 0)

private def cselectE (condition : Sig) (thenValue elseValue : CplxE) : CplxE :=
  (selectE condition thenValue.1 elseValue.1,
   selectE condition thenValue.2 elseValue.2)

private def onlyWhen (condition : Sig) (value : CplxE) : CplxE :=
  cselectE condition value (natE 0)

/-- The same fixed-phase complex carrier used by the simple routed rows. -/
private def poleCarrierE (pole : CplxE) (dSec clkRel : Sig) : CplxE :=
  let incr := div (mul (div pole.2 twoPiE) (lit 4294967296)) .sampleRate
  let phaseQ := modePhaseQFromIncr (toIntE incr) clkRel
  let carrierCos := div (toFloatE (fixedCosCycSig phaseQ)) (lit 1073741824)
  let carrierSin := div (toFloatE (fixedSinCycSig phaseQ)) (lit 1073741824)
  cscaleE (expSig (mul pole.1 dSec)) (carrierCos, carrierSin)

/-- Stable first divided difference of the exponential carrier:
    `(exp(lam*t) - exp(nu*t)) / (lam - nu)`. -/
private def dividedDifferenceWithCarrierE (lam nu nuCarrier : CplxE)
    (dSec clkRel : Sig) : CplxE :=
  let delta := csubE lam nu
  let z := cscaleE dSec delta
  let zsq := add (mul z.1 z.1) (mul z.2 z.2)
  let directLane := gt zsq (litF 0.01)
  let zsafe : CplxE :=
    (selectE directLane z.1 (lit 1), selectE directLane z.2 (lit 0))
  let deltaIncr := div (mul (div delta.2 twoPiE) (lit 4294967296)) .sampleRate
  let deltaPhase := modePhaseQFromIncr (toIntE deltaIncr) clkRel
  let deltaCos := div (toFloatE (fixedCosCycSig deltaPhase)) (lit 1073741824)
  let deltaSin := div (toFloatE (fixedSinCycSig deltaPhase)) (lit 1073741824)
  let ezScale := expSig (mul delta.1 dSec)
  let ez : CplxE := (mul ezScale deltaCos, mul ezScale deltaSin)
  let direct := cdivE (csubE ez (natE 1)) zsafe
  let series := cexpm1SeriesE z
  let exprel := cselectE directLane direct series
  cmulE nuCarrier (cscaleE dSec exprel)

private def dividedDifferenceCarrierE (lam nu : CplxE)
    (dSec clkRel : Sig) : CplxE :=
  dividedDifferenceWithCarrierE lam nu (poleCarrierE nu dSec clkRel) dSec clkRel

/-- Bivariate series for the second divided difference around the third pole.
    With `x=(lam-r)*t` and `y=(p-r)*t`, the dimensionless factor is
    `sum x^i*y^j/(i+j+2)!`.  Eight total degrees leave ample f32 headroom in
    the `|x|,|y| < 0.1` lane selected below.  The complete homogeneous
    polynomial recurrence `h[d] = x*h[d-1] + y^d` shares every power instead
    of respelling each monomial. -/
private def secondExprelSeriesE (x y : CplxE) : CplxE := Id.run do
  let mut homogeneous := natE 1
  let mut yPower := natE 1
  let mut total := cscaleE (litF 0.5) homogeneous
  for degree in [1:9] do
    yPower := cmulE yPower y
    homogeneous := caddE (cmulE x homogeneous) yPower
    total := caddE total (cscaleE
      (div (lit 1) (lit (Int.ofNat (factorial (degree + 2))))) homogeneous)
  return total

/-- Stable second divided difference for three future simple poles.  Direct
    nested DD formulas cover separated scaled poles.  The bivariate series
    covers the small-time cancellation lens and the exact triple collision;
    every inactive direct denominator is replaced before eager evaluation. -/
private def secondDividedDifferenceWithFirstE (lam p r : CplxE)
    (dSec clkRel : Sig) (ddLamP ddLamR rCarrier : CplxE) : CplxE :=
  let delta1 := csubE lam p
  let delta2 := csubE lam r
  let inner1 := sourceCrossingInner delta1
  let inner2 := sourceCrossingInner delta2
  let x := cscaleE dSec delta2
  let y := cscaleE dSec (csubE p r)
  let xSmall := Sig.binary .lt
    (add (mul x.1 x.1) (mul x.2 x.2)) (litF 0.01)
  let ySmall := Sig.binary .lt
    (add (mul y.1 y.1) (mul y.2 y.2)) (litF 0.01)
  let bothInner := Sig.binary .and inner1 inner2
  let seriesLane := Sig.binary .or bothInner (Sig.binary .and xSmall ySmall)
  -- First divided differences are symmetric.  One authored room/room DD is
  -- therefore sufficient for both stable direct formulas, and the two
  -- source/room DDs are shared with the pair corrections at the call site.
  let ddPR := dividedDifferenceCarrierE p r dSec clkRel
  let safeDelta2 : CplxE :=
    (selectE inner2 (lit 1) delta2.1, selectE inner2 (lit 0) delta2.2)
  let directAtP := cdivE (csubE ddLamP ddPR) safeDelta2
  let safeDelta1 : CplxE :=
    (selectE inner1 (lit 1) delta1.1, selectE inner1 (lit 0) delta1.2)
  let directAtR := cdivE (csubE ddLamR ddPR) safeDelta1
  let direct := cselectE inner1 directAtP
    (cselectE inner2 directAtR directAtP)
  let series := cmulE rCarrier
    (cscaleE (mul dSec dSec) (secondExprelSeriesE x y))
  cselectE seriesLane series direct

private def secondDividedDifferenceCarrierE (lam p r : CplxE)
    (dSec clkRel : Sig) : CplxE :=
  let pCarrier := poleCarrierE p dSec clkRel
  let rCarrier := poleCarrierE r dSec clkRel
  let ddLamP := dividedDifferenceWithCarrierE lam p pCarrier dSec clkRel
  let ddLamR := dividedDifferenceWithCarrierE lam r rCarrier dSec clkRel
  secondDividedDifferenceWithFirstE lam p r dSec clkRel
    ddLamP ddLamR rCarrier

/-- One source row carries both its ordinary factored amplitude and any
    source/room confluence correction.  Sharing the source carrier here avoids
    a second mapped phase and a duplicate fixed-phase evaluation. -/
private def routedSourceSideSig (factored : FactoredTwoRoomTerminal)
    (clkInt anchorSamples : Sig) : Sig × Sig :=
  let capacity := factored.source.size
  if capacity == 0 then (lit 0, lit 0) else
  let sourceCols := modeColumns factored.source
  let sourceIndex := Sig.loopIdx 15
  let lam := sourceCols.readPole sourceIndex
  let sourceAmp := factored.sourceAmps.readDynamic
    factored.sourceAssembly sourceIndex
  let p := readImageParts factored.confluenceImages
    factored.confluencePole1 sourceIndex
  let r := readImageParts factored.confluenceImages
    factored.confluencePole2 sourceIndex
  let pair1DDCoeff := readImageParts factored.confluenceImages
    factored.confluencePair1DD sourceIndex
  let pair1SimpleCoeff := readImageParts factored.confluenceImages
    factored.confluencePair1Simple sourceIndex
  let pair2DDCoeff := readImageParts factored.confluenceImages
    factored.confluencePair2DD sourceIndex
  let pair2SimpleCoeff := readImageParts factored.confluenceImages
    factored.confluencePair2Simple sourceIndex
  let tripleCoeff := readImageParts factored.confluenceImages
    factored.confluenceTriple sourceIndex
  let hot1 := gt (readScalarParts factored.confluenceImages
    factored.confluenceHot1Offset sourceIndex) (lit 0)
  let hot2 := gt (readScalarParts factored.confluenceImages
    factored.confluenceHot2Offset sourceIndex) (lit 0)
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let sourceCarrier := poleCarrierE lam dSec clkRel
  let pCarrier := poleCarrierE p dSec clkRel
  let rCarrier := poleCarrierE r dSec clkRel
  -- Sum every coefficient sharing the source carrier before multiplying it.
  -- Near a source/room crossing these terms can be individually large and
  -- cancel; multiplying them separately spends float32 precision twice.
  let simpleCoeff := caddE sourceAmp
    (caddE (onlyWhen hot1 pair1SimpleCoeff)
      (onlyWhen hot2 pair2SimpleCoeff))
  let simple := cmulE simpleCoeff sourceCarrier
  let dd1 := dividedDifferenceWithCarrierE lam p pCarrier dSec clkRel
  let pair1 := onlyWhen hot1 (cmulE pair1DDCoeff dd1)
  let dd2 := dividedDifferenceWithCarrierE lam r rCarrier dSec clkRel
  let pair2 := onlyWhen hot2 (cmulE pair2DDCoeff dd2)
  let tripleCarrier := secondDividedDifferenceWithFirstE lam p r dSec clkRel
    dd1 dd2 rCarrier
  let triple := onlyWhen (Sig.binary .or hot1 hot2)
    (cmulE tripleCoeff tripleCarrier)
  let side := caddE simple (caddE (caddE pair1 pair2) triple)
  let strike := caddE (onlyWhen hot1 pair1SimpleCoeff)
    (onlyWhen hot2 pair2SimpleCoeff)
  let values := #[selectE (gt clkRel (lit 0)) side.1 (lit 0), strike.1]
  let routes := (Array.range capacity).foldl
    (fun routes _ => routes.push (some 0) |>.push (some 1)) #[]
  let tables := sourceCols.tables ++ #[factored.sourceAssembly] ++
    factored.confluenceImages
  let image := Sig.routedSum capacity 2 routes tables values none 15
  (Sig.index image (lit 0), Sig.index image (lit 1))

/-- The structurally cold source lane has no DD or collision correction.  Do
    not merely feed zero confluence images through `routedSourceSideSig`: the
    symbolic backend is eager, so spelling the unreachable DD carrier would
    still consume threadgroup scalar scratch. -/
private def routedColdSourceSideSig (factored : FactoredTwoRoomTerminal)
    (clkInt anchorSamples : Sig) : Sig × Sig :=
  let capacity := factored.source.size
  if capacity == 0 then (lit 0, lit 0) else
  let sourceCols := modeColumns factored.source
  let sourceIndex := Sig.loopIdx 15
  let lam := sourceCols.readPole sourceIndex
  let sourceAmp := factored.sourceAmps.readDynamic
    factored.sourceAssembly sourceIndex
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let side := cmulE sourceAmp (poleCarrierE lam dSec clkRel)
  let value := selectE (gt clkRel (lit 0)) side.1 (lit 0)
  let image := Sig.routedSum capacity 1 (sumRoutes capacity)
    (sourceCols.tables ++ #[factored.sourceAssembly]) #[value] none 15
  (Sig.index image (lit 0), lit 0)

/-- Fuse both ordinary room carriers and their paired DD row into one routed
    phase.  Besides removing two barrier triplets per side, the room-2 carrier
    is shared with the DD evaluation. -/
private def routedRoomSideSig (room1 room2 : Array ModalMode) (assembly : Sig)
    (coeff1 coeff2 coeffDD : FactoredSlice) (clkInt anchorSamples : Sig)
    (binder : Nat) : Sig :=
  if room1.isEmpty then lit 0 else
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let leftCols := modeColumns room1
  let rightCols := modeColumns room2
  let index := Sig.loopIdx binder
  let p := leftCols.readPole index
  let r := rightCols.readPole index
  let carrier1 := poleCarrierE p dSec clkRel
  let carrier2 := poleCarrierE r dSec clkRel
  let dd := dividedDifferenceWithCarrierE p r carrier2 dSec clkRel
  let ordinary1 := cmulE (coeff1.readDynamic assembly index) carrier1
  let ordinary2 := cmulE (coeff2.readDynamic assembly index) carrier2
  let paired := cmulE (coeffDD.readDynamic assembly index) dd
  let value := (caddE (caddE ordinary1 ordinary2) paired).1
  let image := Sig.routedSum room1.size 1 (sumRoutes room1.size)
    (leftCols.tables ++ rightCols.tables ++ #[assembly]) #[value] none binder
  selectE (gt clkRel (lit 0)) (Sig.index image (lit 0)) (lit 0)

def FactoredTwoRoomTerminal.realizeSig (factored : FactoredTwoRoomTerminal)
    (clkInt anchorSamples : Sig) : Sig :=
  let anchorQ := toIntE (mul anchorSamples (lit 4294967296))
  let mirroredClock := sub (mul (lit 2) anchorQ) clkInt
  let source := if factored.confluenceImages.isEmpty then
      routedColdSourceSideSig factored clkInt anchorSamples
    else routedSourceSideSig factored clkInt anchorSamples
  let future := routedRoomSideSig factored.room1 factored.room2
    factored.roomAssembly factored.roomFuture1 factored.roomFuture2
    factored.roomFutureDD clkInt anchorSamples 8
  let past := routedRoomSideSig factored.room1 factored.room2
    factored.roomAssembly factored.roomPast1 factored.roomPast2
    factored.roomPastDD mirroredClock anchorSamples 9
  let sides := add source.1 (add future past)
  let atStrike := Sig.binary .eq (relClockQ clkInt anchorSamples) (lit 0)
  selectE atStrike (add factored.atZero.1 source.2) sides

/-- Assemble the exact two-room degree-zero terminal from shared routed images.
    `none` is a structural refusal and tells the caller to retain the generic
    scalar `Bank` path. -/
def factoredTwoRoomTerminal? (source room1 room2 : Array ModalMode)
    (direction1 direction2 : Sig) : Option FactoredTwoRoomTerminal := do
  if !factoredTwoRoomAdmitted source room1 room2 then none
  let sourceImage ← sourceRoomImage? source room1 room2
  let difference ← differenceImage? room1 room2
  let physical ← physicalImage? room1 room2
  let confluence ← if sourceRoomConfluenceCold source room1 room2 then
      let layout := ConfluenceLayout.create source.size
      some { images := #[], layout }
    else
      confluenceAssemblyImage? source room1 room2 sourceImage difference physical
        direction1 direction2
  let sourceAssembly ← sourceAssemblyImage? source sourceImage direction1 direction2
  let roomAssembly ← roomAssemblyImage? room1 room2 sourceImage difference physical
    direction1 direction2
  let atZero := caddE
    (sourceAssembly.layout.strike.read sourceAssembly.image 0)
    (roomAssembly.layout.strike.read roomAssembly.image 0)
  pure {
    source, room1, room2
    direction1, direction2
    confluenceImages := confluence.images
    confluencePair1DD := confluence.layout.pair1DD
    confluencePair1Simple := confluence.layout.pair1Simple
    confluencePair2DD := confluence.layout.pair2DD
    confluencePair2Simple := confluence.layout.pair2Simple
    confluenceTriple := confluence.layout.triple
    confluencePole1 := confluence.layout.pole1
    confluencePole2 := confluence.layout.pole2
    confluenceHot1Offset := confluence.layout.hot1Offset
    confluenceHot2Offset := confluence.layout.hot2Offset
    sourceAssembly := sourceAssembly.image
    sourceAmps := sourceAssembly.layout.amps
    roomAssembly := roomAssembly.image
    roomFuture1 := roomAssembly.layout.future1
    roomFuture2 := roomAssembly.layout.future2
    roomPast1 := roomAssembly.layout.past1
    roomPast2 := roomAssembly.layout.past2
    roomFutureDD := roomAssembly.layout.futureDD
    roomPastDD := roomAssembly.layout.pastDD
    atZero }

/-! ## Six-section product extension

The phaser-pole rows are structurally outside every room-frequency confluence
lens.  The product schedule therefore folds their room-indexed contributions
into the incumbent source/room images, retains one compact 4-complex source
summary for those rows, and publishes their live poles/residues once through a
24-float cooperative image.  Persistent scratch grows by only those summaries
and the six additional source amplitudes; room-pair and room-assembly images
remain shared. -/

private def inlineModeField (modes : Array ModalMode) (index : Sig)
    (field : ModalMode → CplxE) : CplxE :=
  modes.zipIdx.foldl (fun selected (mode, i) =>
    cselectE (Sig.binary .eq index (litI (Int.ofNat i))) (field mode) selected)
    (natE 0)

/-- Evaluate all live phaser rows once in one small cooperative image.  The
    interleaved layout avoids serial scalar scratch and lets every larger map
    use indexed reads instead of respelling six-way selections. -/
private def liveModeImage? (modes : Array ModalMode) : Option Sig := do
  if modes.isEmpty then none
  let index := Sig.loopIdx 16
  let pole := inlineModeField modes index (fun mode => mode.poleE)
  let amp := inlineModeField modes index (fun mode => mode.ampE)
  let values := #[pole.1, pole.2, amp.1, amp.2]
  let routes := (Array.range modes.size).foldl (fun routes i =>
    routes.push (some (4 * i)) |>.push (some (4 * i + 1))
      |>.push (some (4 * i + 2)) |>.push (some (4 * i + 3))) #[]
  routedImage? modes.size (4 * modes.size) routes #[] values 16

private def liveModePole (image index : Sig) : CplxE :=
  (Sig.index image (mul (lit 4) index),
   Sig.index image (add (lit 1) (mul (lit 4) index)))

private def liveModeAmp (image index : Sig) : CplxE :=
  (Sig.index image (add (lit 2) (mul (lit 4) index)),
   Sig.index image (add (lit 3) (mul (lit 4) index)))

private structure ColdSourceLayout where
  d1 : FactoredSlice
  p1 : FactoredSlice
  d2 : FactoredSlice
  p2 : FactoredSlice
  floatCount : Nat

private def ColdSourceLayout.create (n : Nat) : ColdSourceLayout :=
  let d1 := { name := "phaser-D[0]", offset := 0, extent := n : FactoredSlice }
  let p1 : FactoredSlice := { name := "phaser-P[0]", offset := d1.endOffset, extent := n }
  let d2 : FactoredSlice := { name := "phaser-D[1]", offset := p1.endOffset, extent := n }
  let p2 : FactoredSlice := { name := "phaser-P[1]", offset := d2.endOffset, extent := n }
  { d1, p1, d2, p2, floatCount := p2.endOffset }

private structure ColdSourceSummary where
  image : Sig
  layout : ColdSourceLayout

private def ColdSourceSummary.read (summary : ColdSourceSummary)
    (slice : ColdSourceLayout → FactoredSlice) (index : Sig) : CplxE :=
  (slice summary.layout).readDynamic summary.image index

/-- Add the cold phaser rows only to the room-indexed columns of each incumbent
    source/room chunk.  After removing the redundant hot-row `X` columns, the
    combined `12×8×22` mapped-record shape is exactly the existing maximum
    physical-room route arena (`2112` records). -/
private def sourceRoomPhaserImage? (source phaserRows room1 room2 : Array ModalMode)
    (phaserImage : Sig) : Option SourceRoomImage := do
  let n := source.size
  let k := phaserRows.size
  let m := room1.size
  if n == 0 || k == 0 || m == 0 || room2.size != m then none
  let sourceCols := modeColumns source
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let tables := sourceCols.tables ++ room1Cols.tables ++ room2Cols.tables ++
    #[phaserImage]
  let mut parts := #[]
  for chunk in [0:4] do
    let start := chunk * m / 4
    let stop := (chunk + 1) * m / 4
    if start < stop then
      let width := stop - start
      let capacity := (n + k) * width
      let layout := SourceRoomLayout.create n width
      if !layout.valid then none
      let index := Sig.loopIdx 2
      let row := Sig.binary .floorDiv index (litI (Int.ofNat width))
      let roomLocal := Sig.binary .mod index (litI (Int.ofNat width))
      let roomIndex := add roomLocal (litI (Int.ofNat start))
      let original := Sig.binary .lt row (litI (Int.ofNat n))
      let sourceIndex := selectE original row (litI 0)
      let phaserIndex := selectE original (litI 0)
        (sub row (litI (Int.ofNat n)))
      let lam := cselectE original (sourceCols.readPole sourceIndex)
        (liveModePole phaserImage phaserIndex)
      let amp := cselectE original (sourceCols.readAmp sourceIndex)
        (liveModeAmp phaserImage phaserIndex)
      let pole1 := room1Cols.readPole roomIndex
      let residue1 := room1Cols.readAmp roomIndex
      let pole2 := room2Cols.readPole roomIndex
      let residue2 := room2Cols.readAmp roomIndex
      let q1 := cnegE pole1
      let q2 := cnegE pole2
      let delta1 := csubE lam pole1
      let delta2 := csubE lam pole2
      let hot1 := sourceCrossingHot delta1
      let hot2 := sourceCrossingHot delta2
      let anyHot := Sig.binary .or hot1 hot2
      let u1 := safeInverseFor delta1 hot1
      let v1 := cinvSharedE (csubE q1 lam)
      let u2 := safeInverseFor delta2 hot2
      let v2 := cinvSharedE (csubE q2 lam)
      let d1 := unlessHot hot1 (cmulE residue1 u1)
      let p1 := cmulE residue1 v1
      let e1 := unlessHot hot1 (cnegE (cmulE amp u1))
      let e1Cold2 := unlessHot hot2 e1
      let m1 := cmulE amp v1
      let d2 := unlessHot hot2 (cmulE residue2 u2)
      let p2 := cmulE residue2 v2
      let e2 := unlessHot hot2 (cnegE (cmulE amp u2))
      let m2 := cmulE amp v2
      let tf := unlessHot anyHot (cmulE amp (cmulE u1 u2))
      let tp := cmulE amp (cmulE v1 v2)
      let values := #[d1.1, d1.2, p1.1, p1.2,
        e1.1, e1.2, e1Cold2.1, e1Cold2.2, m1.1, m1.2,
        d2.1, d2.2, p2.1, p2.2, e2.1, e2.2, m2.1, m2.2,
        tf.1, tf.2, tp.1, tp.2]
      let routes := (Array.range capacity).foldl (fun routes item =>
        let sourceRow := item / width
        let j := item % width
        let routes := if sourceRow < n then
            appendComplexRoute
              (appendComplexRoute routes layout.d1 sourceRow) layout.p1 sourceRow
          else routes.push none |>.push none |>.push none |>.push none
        let routes := appendComplexRoute
          (appendComplexRoute
            (appendComplexRoute routes layout.e1 j) layout.e1Cold2 j) layout.m1 j
        let routes := if sourceRow < n then
            appendComplexRoute
              (appendComplexRoute routes layout.d2 sourceRow) layout.p2 sourceRow
          else routes.push none |>.push none |>.push none |>.push none
        let routes := appendComplexRoute
          (appendComplexRoute
            (appendComplexRoute
              (appendComplexRoute routes layout.e2 j) layout.m2 j) layout.tf j)
          layout.tp j
        routes) #[]
      let image ← routedImage? capacity layout.floatCount routes tables values 2
      parts := parts.push { image, layout, roomStart := start }
  let _ ← parts[0]?
  pure { parts }

/-- One reused mapped region computes the four room transfer summaries needed
    to assemble all six phaser-pole source residues. -/
private def coldSourceSummaryImage? (phaserRows room1 room2 : Array ModalMode)
    (phaserImage : Sig) : Option ColdSourceSummary := do
  let k := phaserRows.size
  let m := room1.size
  if k == 0 || m == 0 || room2.size != m then none
  let layout := ColdSourceLayout.create k
  let room1Cols := modeColumns room1
  let room2Cols := modeColumns room2
  let index := Sig.loopIdx 10
  let sourceIndex := Sig.binary .floorDiv index (litI (Int.ofNat m))
  let roomIndex := Sig.binary .mod index (litI (Int.ofNat m))
  let lam := liveModePole phaserImage sourceIndex
  let pole1 := room1Cols.readPole roomIndex
  let pole2 := room2Cols.readPole roomIndex
  let residue1 := room1Cols.readAmp roomIndex
  let residue2 := room2Cols.readAmp roomIndex
  let d1 := cmulE residue1 (cinvSharedE (csubE lam pole1))
  let p1 := cmulE residue1 (cinvSharedE (csubE (cnegE pole1) lam))
  let d2 := cmulE residue2 (cinvSharedE (csubE lam pole2))
  let p2 := cmulE residue2 (cinvSharedE (csubE (cnegE pole2) lam))
  let values := #[d1.1, d1.2, p1.1, p1.2, d2.1, d2.2, p2.1, p2.2]
  let routes := (Array.range (k * m)).foldl (fun routes item =>
    let i := item / m
    appendComplexRoute
      (appendComplexRoute
        (appendComplexRoute
          (appendComplexRoute routes layout.d1 i) layout.p1 i) layout.d2 i)
      layout.p2 i) #[]
  let image ← routedImage? (k * m) layout.floatCount routes
    (room1Cols.tables ++ room2Cols.tables ++ #[phaserImage]) values 10
  pure { image, layout }

private def phaserSourceAssemblyImage? (source phaserRows : Array ModalMode)
    (sourceRoom : SourceRoomImage) (cold : ColdSourceSummary)
    (phaserImage direction1 direction2 : Sig) :
    Option (AssemblyImage SourceAssemblyLayout) := do
  let n := source.size
  let k := phaserRows.size
  if n == 0 || k == 0 then none
  let layout := SourceAssemblyLayout.create (n + k)
  let sourceCols := modeColumns source
  let index := Sig.loopIdx 5
  let original := Sig.binary .lt index (litI (Int.ofNat n))
  let sourceIndex := selectE original index (litI 0)
  let phaserIndex := selectE original (litI 0)
    (sub index (litI (Int.ofNat n)))
  let chooseSummary := fun
      (sourceSlice : SourceRoomLayout → FactoredSlice)
      (coldSlice : ColdSourceLayout → FactoredSlice) =>
    cselectE original (sourceRoom.readSource sourceSlice sourceIndex)
      (cold.read coldSlice phaserIndex)
  let gain1 := caddE
    (cscaleE (sub (lit 1) direction1) (chooseSummary (fun l => l.d1) (fun l => l.d1)))
    (cscaleE direction1 (chooseSummary (fun l => l.p1) (fun l => l.p1)))
  let gain2 := caddE
    (cscaleE (sub (lit 1) direction2) (chooseSummary (fun l => l.d2) (fun l => l.d2)))
    (cscaleE direction2 (chooseSummary (fun l => l.p2) (fun l => l.p2)))
  let rawAmp := cselectE original (sourceCols.readAmp sourceIndex)
    (liveModeAmp phaserImage phaserIndex)
  let amp := cmulE rawAmp (cmulE gain1 gain2)
  let values := #[amp.1, amp.2, amp.1, amp.2]
  let routes := (Array.range (n + k)).foldl (fun routes i =>
    appendComplexRoute (appendComplexRoute routes layout.amps i) layout.strike 0) #[]
  let image ← routedImage? (n + k) layout.floatCount routes
    (sourceRoom.images ++ #[cold.image, phaserImage] ++ sourceCols.tables) values 5
  pure { image, layout }

/-- Exact fused terminal for `source → room → six-section phaser → room`.
    The caller has already proved the frozen-LTI commutation that moves the
    phaser beside the source for this terminal coordinate. -/
structure FactoredTwoRoomPhaserTerminal where
  base : FactoredTwoRoomTerminal
  phaserRows : Array ModalMode
  phaserImage : Sig
  sourceAssembly : Sig
  phaserAmps : FactoredSlice

private def routedInlineColdSourceSideSig (modes : Array ModalMode)
    (modeImage assembly : Sig) (amps : FactoredSlice)
    (clkInt anchorSamples : Sig) : Sig :=
  let capacity := modes.size
  if capacity == 0 then lit 0 else
  let index := Sig.loopIdx 15
  let pole := liveModePole modeImage index
  let amp := amps.readDynamic assembly index
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let side := cmulE amp (poleCarrierE pole dSec clkRel)
  let image := Sig.routedSum capacity 1 (sumRoutes capacity)
    #[modeImage, assembly]
    #[selectE (gt clkRel (lit 0)) side.1 (lit 0)] none 15
  Sig.index image (lit 0)

def FactoredTwoRoomPhaserTerminal.realizeSig
    (factored : FactoredTwoRoomPhaserTerminal)
    (clkInt anchorSamples : Sig) : Sig :=
  add (factored.base.realizeSig clkInt anchorSamples)
    (routedInlineColdSourceSideSig factored.phaserRows factored.phaserImage
      factored.sourceAssembly factored.phaserAmps clkInt anchorSamples)

def factoredTwoRoomPhaserTerminal? (source tails room1 room2 : Array ModalMode)
    (mix direction1 direction2 : Sig) : Option FactoredTwoRoomPhaserTerminal := do
  if tails.isEmpty || !factoredTwoRoomAdmitted source room1 room2 then none
  let decorated := decorateDegreeZeroCausalPhaser source tails mix
  let phaserRows := decorated.extract source.size decorated.size
  if phaserRows.size != tails.size ||
      !sourceRoomConfluenceCold phaserRows room1 room2 then none
  let phaserImage ← liveModeImage? phaserRows
  let sourceImage ← sourceRoomPhaserImage? source phaserRows room1 room2 phaserImage
  let cold ← coldSourceSummaryImage? phaserRows room1 room2 phaserImage
  let difference ← differenceImage? room1 room2
  let physical ← physicalImage? room1 room2
  let confluence ← confluenceAssemblyImage? source room1 room2 sourceImage
    difference physical direction1 direction2
  let sourceAssembly ← phaserSourceAssemblyImage? source phaserRows sourceImage
    cold phaserImage direction1 direction2
  let roomAssembly ← roomAssemblyImage? room1 room2 sourceImage difference physical
    direction1 direction2
  let atZero := caddE
    (sourceAssembly.layout.strike.read sourceAssembly.image 0)
    (roomAssembly.layout.strike.read roomAssembly.image 0)
  let base : FactoredTwoRoomTerminal := {
    source, room1, room2
    direction1, direction2
    confluenceImages := confluence.images
    confluencePair1DD := confluence.layout.pair1DD
    confluencePair1Simple := confluence.layout.pair1Simple
    confluencePair2DD := confluence.layout.pair2DD
    confluencePair2Simple := confluence.layout.pair2Simple
    confluenceTriple := confluence.layout.triple
    confluencePole1 := confluence.layout.pole1
    confluencePole2 := confluence.layout.pole2
    confluenceHot1Offset := confluence.layout.hot1Offset
    confluenceHot2Offset := confluence.layout.hot2Offset
    sourceAssembly := sourceAssembly.image
    sourceAmps := { sourceAssembly.layout.amps with extent := source.size }
    roomAssembly := roomAssembly.image
    roomFuture1 := roomAssembly.layout.future1
    roomFuture2 := roomAssembly.layout.future2
    roomPast1 := roomAssembly.layout.past1
    roomPast2 := roomAssembly.layout.past2
    roomFutureDD := roomAssembly.layout.futureDD
    roomPastDD := roomAssembly.layout.pastDD
    atZero }
  let phaserAmps : FactoredSlice := {
    name := "phaser-source-future"
    offset := 2 * source.size
    extent := phaserRows.size }
  pure {
    base := base
    phaserRows := phaserRows
    phaserImage := phaserImage
    sourceAssembly := sourceAssembly.image
    phaserAmps := phaserAmps }

end Tropical.EmitArrow.Oriented
