module Runtime where

import Foreign.C.Types
import Foreign.Ptr

import qualified Expression as Expr
import Object

import qualified Data.Set as Set

data Buffer

data Runtime = Runtime
    { env :: Set Variable
    , output :: Ptr Buffer
    }
    deriving (Show)

data CExpr
foreign import ccall safe "makeExternal" makeExternal 
    :: Ptr Signal -> IO (Ptr CExpr)
foreign import ccall safe "makePlus" makePlus 
    :: Ptr CExpr -> Ptr CExpr -> IO (Ptr CExpr)
foreign import ccall safe "makeTimes" makeTimes 
    :: Ptr CExpr -> Ptr CExpr -> IO (Ptr CExpr)
foreign import ccall safe "makeNegate" makeNegate 
    :: Ptr CExpr -> IO (Ptr CExpr)
foreign import ccall safe "makeLiteral" makeLiteral 
    :: Signal -> IO (Ptr CExpr)

-- only used for testing, normally a CExpr should
-- be evaluated by the C++ runtime
foreign import ccall safe "eval" evalc 
    :: Ptr CExpr -> IO Signal

data CRuntime
foreign import ccall unsafe "compute" computeC 
    :: Ptr CRuntime -> IO ()

{--

setting assignments

verification
    lhs 
        variable
            check to see if variable in dictionary
        object input:
            check for object in dictionary
            check whether object has specified input
            check type lhs == type rhs     
    rhs
        variable
            check for expression in dictionary
        object output
            check for object in dictionary
            check whether object has specified output
        expression
            type check
            verify all subexpressions, set pointers
update
    generate runtime representation, capture pointer
    update Haskell runtime
--}