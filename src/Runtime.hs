{-# LANGUAGE TemplateHaskell #-}

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
import Control.Lens

import Sound.PortAudio
import Sound.PortAudio.Base

import Expression
import Object
import Interface 

import Foreign.C.Types
import Foreign.Storable
import Foreign.ForeignPtr
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Marshal.Array

import qualified Data.Set as Set
import qualified Data.Vector as V

data Buffer

data Runtime = Runtime
  { _env :: Set.Set Var
  , _runtime :: Ptr CRuntime
  }
  deriving (Show)

makeLenses ''Runtime

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
  Ref r -> makeExternal undefined
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