{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

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

--import Data.Data hiding (DataType)
--import Data.Generics.Schemes
import qualified Data.Map as Map
import Data.Typeable

import Foreign.C.Types
import Foreign.ForeignPtr
import Foreign.Ptr

import Sound.PortAudio
import Sound.PortAudio.Base

import System.Console.Haskeline

-- Internal imports

import Interface
import qualified Parser as P
--import Parser


data Literal a where 
  BoolLit :: Bool -> Literal BoolType 

instance (Show a) => Show (Literal a) where
  show (BoolLit x) = show x


data BinOp a where
  Plus 
    :: (DataType a, Additive a)
    => Expr a
    -> Expr a 
    -> BinOp a
  Times
    :: (DataType a, Multiplicative a)
    => Expr a
    -> Expr a 
    -> BinOp a

data Expr a where
  Lit
    :: (DataType a) 
    => Literal a 
    -> Expr a

  Ref
    :: (DataType a) 
    => Name
    -> Ptr a
    -> Expr a

class Additive a
class Multiplicative a
class DataType a

-- Types of valid Tropical expressions

data RealType = RealType
  deriving (Show)
instance DataType RealType
instance Additive RealType
instance Multiplicative RealType

data IntType = IntType
  deriving (Show)
instance DataType IntType
instance Additive IntType
instance Multiplicative IntType

data BoolType = BoolType
  deriving (Show)
instance DataType BoolType

-- Types for interfacing with C code

data CRealLiteral = CRealLiteral
instance DataType CRealLiteral

data CIntLiteral = CIntLiteral
instance DataType CIntLiteral

data CBoolLiteral = CBoolLiteral
instance DataType CBoolLiteral

class Direction a
data Input = Input
instance Direction Input
data Output = Output
instance Direction Output

data Object where 
  Module 
    :: Module 
    -> Ptr CModule 
    -> Object
  Expression 
    :: Expr a 
    -> Ptr a
    -> Object
  --deriving (Show)

-- Port has a type and belongs to an object
-- Can be input or output
data Port a b where
  Port
    :: (DataType a, Direction b)
    => Name
    -> (Ptr a) 
    -> Port a b
  --deriving (Show)

data VCO =
  VCO
    { baseFreq :: Expr RealType
    , fm :: Expr RealType
    , basePhase :: Expr RealType
    , pm :: Expr RealType
    }
  --deriving (Show)

data Module
  = VCOModule VCO
  --deriving (Show)


data StaticError 
  = TypeError String
  | Undeclared String
  | Invalid String

instance Show StaticError where
  show (TypeError s) = "Type error: " <> s
  show (Undeclared s) = "Undeclared variable: " <> s
  show (Invalid s) = "Invalid field: " <> s


newtype Name = Name { unName :: String } 
  deriving (Eq, Ord, Show)

newtype Env = Env { _objects :: Map.Map Name Object }

makeLenses ''Env
getPort = undefined

resolveRef
  :: DataType a
  => Env
  -> P.Ref 
  -> Either StaticError (Expr a)
resolveRef env ref = case ref of
  P.VarID name ->
    case Map.lookup (Name name) (env^.objects) of
      Nothing -> Left $ Undeclared name
      Just obj -> case obj of
        Module m p -> undefined
        Expression e p -> undefined
  P.Port name field ->
    case Map.lookup (Name name) (env^.objects) of
      Nothing -> Left $ Undeclared name
      Just obj -> case getPort obj of
        Nothing -> Left $ Invalid name
        Just field -> undefined

-- for now this is a stub (everything typechecks)
typeCheck
  :: DataType a
  => P.ParseExpr
  -> Either StaticError (Expr a)
typeCheck = undefined


validateExpr
  :: DataType a
  => Env
  -> P.ParseExpr
  -> Either StaticError (Expr a)
--validateExpr env = resolve env >=> typeCheck
validateExpr = undefined 

data Buffer

data Runtime = Runtime
  { _env :: Env
  , _runtime :: Ptr CRuntime
  }
  --deriving (Show)

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
  :: P.ParseResult 
  -> InputT (StateT Runtime IO) ()
handleParse (P.Show x) = outputStrLn $ show x
handleParse (P.Listen x) = undefined
handleParse (P.Assign ref expr) = undefined
handleParse (P.Create ref obj) = undefined
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
generateCRepr :: Env -> Expr a -> IO (Ptr CExpr)
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