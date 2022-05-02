{-# LANGUAGE TemplateHaskell #-}

module Object where

import Expression
import Foreign.Ptr
import Control.Lens

newtype Freq = Freq { freq :: Double }
  deriving (Show)

newtype FM = FM { fm :: Expr }
  deriving (Show)

data Module
  = VCO Freq FM 
  deriving (Show)

data Object 
  = Module Module
  | Expression Expr
  deriving (Show)

newtype Name = Name { unName :: String } 
  deriving (Eq, Ord, Show)

data CExpr
data CModule
data CObject

data Var = Var 
  { _name :: Name
  , _this :: Ptr CObject
  , _obj :: Object
  }
  deriving (Show)

makeLenses ''Var

instance Eq Var where
  v1 == v2 = v1^.name == v2^.name

instance Ord Var where
  v1 <= v2 = v1^.name <= v2^.name