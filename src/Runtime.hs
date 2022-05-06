{-# LANGUAGE TemplateHaskell #-}

-- This module is responsible for running the parser
-- and analyzer and updating the runtime environment.

module Runtime where

-- Library imports

import Control.Concurrent
import Control.Concurrent.MVar
import Control.Lens
import Control.Monad
import Control.Monad.Trans
import Control.Monad.State

import qualified Data.Set as Set

import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr

import Sound.PortAudio
import Sound.PortAudio.Base

import System.Console.Haskeline

-- Internal imports

import Interface
import Object
import Parser

data StaticError 
  = TypeError String
  | Undeclared String

data CompileExpr

type Env = Set.Set Var

resolve
  :: Env
  -> ParseExpr
  -> Either StaticError CompileExpr
resolve expr = undefined
 
-- for now this is a stub (everything typechecks)
typeCheck
  :: CompileExpr
  -> Either StaticError CompileExpr
typeCheck = Right


validateExpr
  :: Env
  -> ParseExpr
  -> Either StaticError CompileExpr
validateExpr env = resolve env >=> typeCheck

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

handleParse 
  :: ParseResult 
  -> InputT (StateT Runtime IO) ()
handleParse (Show x) =
  outputStrLn $ show x
handleParse (Listen x) = undefined
handleParse (Assign x) = undefined
 {--
  -- here is where type checking should happen
  currentRT <- lift get
  crepr <- liftIO $ generateCRepr (currentRT^.env) result 
  let v = Var 
            { _name = Name ""
            , _this = castPtr crepr
            , _obj = Expression result 
            }
  lift $ put $ over env (Set.insert v) currentRT 
  evaluated <- liftIO $ evalc crepr
  outputStrLn $ show evaluated
--}
generateCRepr :: Env -> Expr -> IO (Ptr CExpr)
generateCRepr = undefined
{--
generateCRepr env expr = case expr of
  Literal s -> makeLiteral s
  Ref r -> makeExternal undefined
  Negate expr -> do
    x <- generateCRepr env expr
    makeNegate x
  Times e1 e2 -> do
    x <- generateCRepr env e1
    y <- generateCRepr env e2
    makeTimes x y
  Plus e1 e2 -> do
    x <- generateCRepr env e1
    y <- generateCRepr env e2
    makePlus x y

sampleAST :: Expr
sampleAST = Times (Plus (Literal 3) (Literal 2)) (Literal 2)
--}