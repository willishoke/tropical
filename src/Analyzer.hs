{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}


-- This module is responsible for static analysis of parser 
-- output, including reference resolution and type checking


module Analyzer where


-- External imports

import Control.Lens
import Control.Monad
import Control.Monad.Reader
import Control.Monad.State
import Control.Monad.Trans
import Control.Monad.Writer

import qualified Data.Map as Map
import Data.Maybe

import Foreign.C.Types
import Foreign.Ptr

import System.Console.Haskeline

-- Internal imports

import Interface
import qualified Parser as P


-- DATATYPES

class DataType a
class (DataType a) => Numeric a

data RealType = RealType
  deriving (Show)
instance DataType RealType 
instance Numeric RealType

data IntType = IntType
  deriving (Show)
instance DataType IntType
instance Numeric IntType

data BoolType = BoolType
  deriving (Show)
instance DataType BoolType

data CValue a where
  CBoolLit :: CValue BoolType
  CIntLit :: CValue IntType
  CRealLit :: CValue RealType

-- LITERALS

data Literal a where 
  BoolLit :: Bool -> Literal BoolType
  IntLit :: Int -> Literal IntType
  RealLit :: Double -> Literal RealType

instance Show (Literal a) where
  show (BoolLit b) = show b
  show (IntLit b) = show b
  show (RealLit b) = show b


-- UNARY OPERATORS

data UnOp a where
  Negate
    :: (Numeric a)
    => Expr a
    -> UnOp a

instance Show (UnOp a) where
  show (Negate expr) = '-' : (show expr)


-- BINARY OPERATORS

data BinOp a where
  Plus 
    :: (Numeric a)
    => Expr a
    -> Expr a 
    -> BinOp a
  Times
    :: (Numeric a)
    => Expr a
    -> Expr a 
    -> BinOp a

instance Show (BinOp a) where
  show (Plus e1 e2) = show e1 <> " + " <> show e2
  show (Times e1 e2) = show e1 <> " * " <> show e2


-- EXPRESSIONS

data Expr a where
  Lit
    :: (DataType a) 
    => Literal a 
    -> Expr a
  Ref
    :: (DataType a)
    => Name
    -> Ptr (CValue a)
    -> Expr a
  BinOp
    :: (DataType a)
    => BinOp a
    -> Expr a
  UnOp
    :: (DataType a) 
    => UnOp a
    -> Expr a

instance Show (Expr a) where
  show (Lit lit) = show lit
  show (Ref name _) = show name
  show (UnOp x) = show x
  show (BinOp x) = show x


-- EXPRESSION CLASSES

data GenericExpr where
  GenericExpr 
    :: (DataType a) 
    => Expr a 
    -> GenericExpr

instance Show GenericExpr where
  show (GenericExpr x) = show x

data NumericExpr where 
  IntExpr :: Expr IntType -> NumericExpr
  RealExpr :: Expr RealType -> NumericExpr

instance Show NumericExpr where
  show (IntExpr x) = show x
  show (RealExpr x) = show x


-- This is used internally by runtime,
-- but access is needed here to resolve references

data StoredExpr where 
  Expr 
    :: (DataType a, Show a)
    => a
    -> Expr a 
    -> Ptr (CValue a)
    -> StoredExpr

instance Show StoredExpr where
  show (Expr t e _) = show e <> " : " <> show t


-- ERROR TYPES

data StaticError 
  = Invalid String
  | TypeError String
  | Undeclared String
  | Unspecified String

instance Show StaticError where
  show (Invalid s) = "Invalid port: " <> s
  show (TypeError s) = "Type error: " <> s
  show (Undeclared s) = "Undeclared variable: " <> s
  show (Unspecified s) = "Unspecified port: " <> s



-- Environment for stored expressions

newtype Name = Name { unName :: String } 
  deriving (Eq, Ord, Show)

newtype Env = Env { _dictionary :: Map.Map Name StoredExpr }
makeLenses ''Env


-- Utility functions

getStoredExpr
  :: Name
  -> Reader Env (Either StaticError StoredExpr)
getStoredExpr name = do
  env <- ask
  case Map.lookup name $ env^.dictionary of 
    Just expr -> pure $ Right expr
    Nothing -> pure $ Left $ Undeclared $ unName name

resolveRef
  :: P.Ref 
  -> Reader Env (Either StaticError GenericExpr)
resolveRef ref = do
  env <- ask 
  expr <- getStoredExpr (Name $ P.unRef ref)
  case expr of
    Right (Expr _ _ p) -> pure $ Right $ GenericExpr (Ref (Name $ P.unRef ref) p)
    Left x -> pure $ Left x


-- Builder functions

buildNumericRef
  :: P.Ref
  -> Reader Env (Either StaticError NumericExpr)
buildNumericRef = undefined

buildNumericUnOp
  :: (forall a. (Numeric a) => Expr a -> UnOp a)
  -> P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
buildNumericUnOp ctr expr = do
  e <- buildNumericExpr expr
  case e of 
    Right (IntExpr ie) -> pure $ Right $ IntExpr $ UnOp $ ctr ie
    Right (RealExpr re) -> pure $ Right $ RealExpr $ UnOp $ ctr re
    Left x -> pure $ Left x

-- Using rank 2 type to build arbitrary numeric data constructor

buildNumericBinOp
  :: (forall a. (Numeric a) => Expr a -> Expr a -> BinOp a)
  -> P.ParseExpr
  -> P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
buildNumericBinOp ctr e1 e2 = do
  e <- buildNumericExpr e1 
  case e of
    Right (IntExpr ie1) -> do
      x <- buildNumericExpr e2
      case x of 
        Right (IntExpr ie2) -> pure $ Right $ IntExpr $ BinOp $ ctr ie1 ie2
        Right _ -> pure $ Left $ TypeError $ "Argument mismatch" 
        Left x -> pure $ Left x
    Right (RealExpr re1) -> do 
      x <- buildNumericExpr e2
      case x of
        Right (RealExpr re2) -> pure $ Right $ RealExpr $ BinOp $ ctr re1 re2
        Right _ -> pure $ Left $ TypeError $ "Argument mismatch" 
        Left x -> pure $ Left x
    Left x -> pure $ Left x

buildNumericExpr
  :: P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
buildNumericExpr expr = case expr of
  P.ParseInt i -> pure $ Right $ IntExpr $ Lit $ IntLit i
  P.ParseReal r -> pure $ Right $ RealExpr $ Lit $ RealLit r
  P.ParseRef r -> undefined
  P.ParseNegate n -> buildNumericUnOp Negate n
  P.ParseTimes e1 e2 -> buildNumericBinOp Times e1 e2 
  P.ParsePlus e1 e2 -> buildNumericBinOp Plus e1 e2 
  other -> pure $ Left $ TypeError $ "Expecting numeric, got " 
                              <> show other

buildBoolExpr 
  :: P.ParseExpr 
  -> Reader Env (Either StaticError (Expr BoolType))
buildBoolExpr expr = case expr of
  P.ParseBool b -> pure $ Right $ Lit $ BoolLit b
  P.ParseRef r -> undefined
  other -> pure $ Left $ TypeError $ "Expecting boolean, got " 
                                     <> show other

buildExpr
  :: P.ParseExpr 
  -> Reader Env (Either StaticError GenericExpr)
buildExpr expr =
  case expr of
    P.ParseBool b -> pure $ Right $ GenericExpr $ Lit $ BoolLit b
    P.ParseInt i -> pure $ Right $ GenericExpr $ Lit $ IntLit i
    P.ParseReal r -> pure $ Right $ GenericExpr $ Lit $ RealLit r
    P.ParseRef pr -> do 
      env <- ask
      case Map.lookup (Name pr) (env^.dictionary) of
        Just _ -> undefined
        Nothing -> undefined
    P.ParseNegate n -> do
      e <- buildNumericUnOp Negate n
      case e of
        Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
        Right (RealExpr re) -> pure $ Right $ GenericExpr re
        Left x -> pure $ Left x
    P.ParsePlus e1 e2 -> do
      e <- buildNumericBinOp Plus e1 e2 
      case e of
        Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
        Right (RealExpr re) -> pure $ Right $ GenericExpr re
        Left x -> pure $ Left x
    P.ParseTimes e1 e2 -> do
      e <- buildNumericBinOp Times e1 e2 
      case e of
        Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
        Right (RealExpr re) -> pure $ Right $ GenericExpr re
        Left x -> pure $ Left x