module Runtime where

import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import Control.Concurrent
import Control.Concurrent.MVar
import Control.Monad

import Sound.PortAudio
import Sound.PortAudio.Base

import Expression
import Object

import qualified Data.Set as Set
import qualified Data.Vector as V

data Buffer

data Runtime = Runtime
    { env :: Set.Set Var
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

foreign import ccall unsafe "computeC" computeC 
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

type Env = Set.Set Var

generateAST :: Env -> Expr -> IO (Ptr CExpr)
generateAST env expr = case expr of
    Literal s -> makeLiteral s
    VarRef v -> makeExternal undefined
    Negate expr -> do
        x <- generateAST env expr
        makeNegate x
    Times e1 e2 -> do
        x <- generateAST env e1
        y <- generateAST env e2
        makeTimes x y
    Plus e1 e2 -> do
        x <- generateAST env e1
        y <- generateAST env e2
        makePlus x y

sampleAST :: Expr
sampleAST = Times (Plus (Literal 3) (Literal 2)) (Literal 2)


sampRate :: Double
sampRate = 44100

framesPerBuffer :: Int
framesPerBuffer = 600


-- blocks until signal received from callback
compute :: Ptr CRuntime -> MVar () -> IO ()
compute runtime readyFlag = forever $ do
  computeC runtime
  takeMVar readyFlag


-- thin wrapper over memcpy
-- signals to compute function when complete
mainCallback :: Ptr CFloat -> MVar () -> StreamCallback CFloat CFloat
mainCallback buffer mvar _ _ frames _ out = do
  let n = fromIntegral frames
  copyArray out buffer n
  putMVar mvar ()
  pure Continue