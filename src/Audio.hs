module Audio where

import Runtime
import Interface

import Foreign.C.Types
import Foreign.Storable
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Marshal.Array

import Control.Concurrent.MVar
import Control.Concurrent
import Control.Monad

import Sound.PortAudio.Base
import Sound.PortAudio


{--
-- audio processing example from Portaudio Haskell bindings at
-- https://github.com/sw17ch/portaudio

numSeconds :: Int
numSeconds = 5

tablesize :: int
tablesize = 200

data SineTable = SineTable { sine :: V.Vector Float }
data Phases = Phases { leftPhase :: Int, rightPhase :: Int }

newTable :: Int -> SineTable
newTable sze = SineTable vec where
  intSze = fromInteger $ toInteger sze
  vec = V.fromList $ map (\i -> sin $ (i / intSze) * pi * 2) [0..(intSze - 1)]

sineTable :: SineTable
sineTable = newTable tableSize

poker :: (Storable a, Fractional a) => Ptr a -> (Int, Int) -> Int -> IO (Int, Int)
poker out (l, r) i = do
  pokeElemOff out (2 * i)      (realToFrac $ (V.!) (sine sineTable) l)
  pokeElemOff out (2 * i + 1)  (realToFrac $ (V.!) (sine sineTable) r)
  let newL = let x = l + 1 in (if x >= tableSize then (x - tableSize) else x)
  let newR = let x = r + 3 in (if x >= tableSize then (x - tableSize) else x)
  return (newL, newR)

paTestCallback :: MVar Phases -> StreamCallback CFloat CFloat
paTestCallback mvar _ _ frames _ out = do
  phases <- readMVar mvar

  (newL', newR') <- foldM (poker out) (leftPhase phases, rightPhase phases) [0..(fromIntegral $ frames - 1)]

  swapMVar mvar (phases { leftPhase = newL', rightPhase = newR' })
  return Continue

--}

-- constants for audio engine
-- TODO: make configurable at runtime

sampRate :: Double
sampRate = 44100

framesPerBuffer :: Int
framesPerBuffer = 600

-- blocks until signal received from callback
compute :: Ptr CRuntime -> MVar () -> IO ()
compute runtime readyFlag = forever $ do
  computeRuntime runtime
  takeMVar readyFlag

-- thin wrapper over memcpy
-- signals to compute thread when complete
mainCallback :: Ptr CFloat -> MVar () -> StreamCallback CFloat CFloat
mainCallback buffer mvar _ _ frames _ out = do
  let n = fromIntegral frames
  copyArray out buffer n
  putMVar mvar ()
  pure Continue