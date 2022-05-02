module Interface where

-- This module is responsible for providing an interface
-- between the Haskell runtime and the C++ runtime

import Foreign.C.Types
import Foreign.Ptr

import Object

data CRuntime

-- Functions for building syntax tree representations in C++

foreign import ccall safe "makeExternal" makeExternal 
  :: Ptr CDouble -> IO (Ptr CExpr)

foreign import ccall safe "makePlus" makePlus 
  :: Ptr CExpr -> Ptr CExpr -> IO (Ptr CExpr)

foreign import ccall safe "makeTimes" makeTimes 
  :: Ptr CExpr -> Ptr CExpr -> IO (Ptr CExpr)

foreign import ccall safe "makeNegate" makeNegate 
  :: Ptr CExpr -> IO (Ptr CExpr)

foreign import ccall safe "makeLiteral" makeLiteral 
  :: CDouble -> IO (Ptr CExpr)

-- Only used for testing, normally a CExpr should
-- be evaluated by the C++ runtime

foreign import ccall safe "eval" evalc 
  :: Ptr CExpr -> IO CDouble


-- Functions for interfacing with C++ runtime

foreign import ccall unsafe "initRuntime" initRuntime
  :: CUInt -> IO (Ptr CRuntime)

foreign import ccall unsafe "deleteRuntime" deleteRuntime
  :: Ptr CRuntime -> IO ()

foreign import ccall unsafe "computeC" computeC 
  :: Ptr CRuntime -> IO ()

foreign import ccall unsafe "addObject" addObject 
  :: Ptr CRuntime -> Ptr CObject -> IO ()

foreign import ccall unsafe "getBufferAddress" getBufferAddress 
  :: Ptr CRuntime -> IO (Ptr CFloat)
