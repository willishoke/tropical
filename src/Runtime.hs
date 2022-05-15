{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE Rank2Types #-}

-- This module is responsible for handling parser output
-- and updating the runtime environment.

module Runtime where

-- Library imports

import Control.Concurrent
import Control.Concurrent.MVar
import Control.Lens
import Control.Monad
import Control.Monad.Reader
import Control.Monad.State
import Control.Monad.Trans
import Control.Monad.Writer

import qualified Data.Map as Map
import Data.Maybe
import Data.Typeable

import Foreign.C.Types
import Foreign.Ptr

import Sound.PortAudio
import Sound.PortAudio.Base

import System.Console.Haskeline

-- Internal imports

import Interface
import qualified Parser as P


data Literal a where 
  BoolLit :: Bool -> Literal BoolType
  IntLit :: Int -> Literal IntType
  RealLit :: Double -> Literal RealType
  deriving (Typeable)

instance Show (Literal a) where
  show (BoolLit b) = show b
  show (IntLit b) = show b
  show (RealLit b) = show b

data CValue a where
  CBoolLit :: CValue BoolType
  CIntLit :: CValue IntType
  CRealLit :: CValue RealType

data UnOp a where
  Negate
    :: (Numeric a)
    => Expr a
    -> UnOp a

instance Show (UnOp a) where
  show (Negate expr) = '-' : (show expr)

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
  deriving (Typeable)

instance Show (BinOp a) where
  show (Plus e1 e2) = show e1 <> " + " <> show e2
  show (Times e1 e2) = show e1 <> " * " <> show e2

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
  deriving (Typeable)

instance Show (Expr a) where
  show (Lit lit) = show lit
  show (Ref name _) = show name
  show (UnOp x) = show x
  show (BinOp x) = show x


-- Types of valid Tropical expressions

class DataType a
class (DataType a) => Numeric a

data RealType = RealType
  deriving (Show, Typeable)
instance DataType RealType 
instance Numeric RealType

data IntType = IntType
  deriving (Show, Typeable)
instance DataType IntType
instance Numeric IntType

data BoolType = BoolType
  deriving (Show, Typeable)
instance DataType BoolType


data GenericExpr where
  GenericExpr 
    :: (DataType a) 
    => Expr a 
    -> GenericExpr

instance Show GenericExpr where
  show (GenericExpr w) = show w

data Object where 
  Module 
    :: Module 
    -> Ptr CModule
    -> Object
  Expr 
    :: (DataType a, Show a)
    => a
    -> Expr a 
    -> Ptr (CValue a)
    -> Object

instance Show Object where
  show (Module m _) = show m
  show (Expr t e _) = show e <> " : " <> show t

-- Port has a type and belongs to an object
-- Can be input or output
data Port a where
  Input
    :: (DataType a) 
    => Expr a 
    -> Port a
  Output 
    :: (DataType a) 
    => Expr a 
    -> Port a

instance Show (Port a) where
  show (Input p) = show p
  show (Output p) = show p

data VCO =
  VCO
    { freq :: Port RealType 
    , phase :: Port RealType 
    }
  deriving (Show)

data Module
  = VCOModule VCO
  deriving (Show)


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

newtype Name = Name { unName :: String } 
  deriving (Eq, Ord, Show)

newtype Env = Env { _objects :: Map.Map Name Object }
makeLenses ''Env


showPort :: String -> Module -> Maybe String
showPort s m = case m of
  VCOModule (VCO f p) -> case s of
    "freq" -> Just $ show f 
    "phase" -> Just $ show p
    _ -> Nothing


getObj 
  :: Name
  -> Reader Env (Either StaticError Object)
getObj n = do
  e <- ask
  case Map.lookup n $ e^.objects of 
    Nothing -> pure $ Left $ Undeclared $ unName n
    Just obj -> pure $ Right obj


resolveRef
  :: P.Ref 
  -> Reader Env (Either StaticError GenericExpr)
resolveRef ref = do
  env <- ask 
  case ref of
    P.Var name -> 
      getObj (Name name) >>= \obj -> case obj of
        Right (Module m p) -> pure $ Left $ Unspecified name
        Right (Expr _ _ p) -> pure $ Right $ GenericExpr (Ref (Name name) p)
        Left x -> pure $ Left x
    P.Port objName portName -> undefined
    {--
          let casts = zipWith cast 
                        (fmap [BoolType, IntType, RealType] (repeat e)
          let option = head $ catMaybes $ 
          case t of
          BoolType -> pure $ Right $ GenericExpr (Ref (Name name) p :: Expr BoolType)
          IntType -> pure $ Right $ GenericExpr (Ref (Name name) p :: Expr IntType)
          RealType -> pure $ Right $ GenericExpr (Ref (Name name) p :: Expr RealType)
      getObj (Name objName) >>= \obj -> case obj of
        Expr _ _ _ -> pure $ Left $ Invalid objName
        Module m _ -> case getPort portName m of
          Nothing -> pure $ Left $ Invalid portName 
          Just (Input _) -> pure $ Right undefined
          Just (Output _) -> pure $ Right undefined
    --}

getPort = undefined

data Buffer

data Runtime = Runtime
  { _env :: Env
  , _runtime :: Ptr CRuntime
  }

makeLenses ''Runtime


nameExists 
  :: P.Ref
  -> Env
  -> Bool
nameExists ref env = case ref of
  P.Var v -> m v env
  P.Port p _ -> m p env
  where m x e = Map.member (Name x) (e^.objects)

buildListen = undefined
buildAssign = undefined
buildObj = undefined

data NumericExpr where 
  IntExpr :: Expr IntType -> NumericExpr
  RealExpr :: Expr RealType -> NumericExpr

handleNumericRef
  :: P.Ref
  -> Reader Env (Either StaticError NumericExpr)
handleNumericRef = undefined

handleNumericUnOp
  :: (forall a. (Numeric a) => Expr a -> UnOp a)
  -> P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
handleNumericUnOp ctr expr = do
  e <- handleNumericExpr expr
  case e of 
    Right (IntExpr ie) -> pure $ Right $ IntExpr $ UnOp $ ctr ie
    Right (RealExpr re) -> pure $ Right $ RealExpr $ UnOp $ ctr re
    Left x -> pure $ Left x

-- Using rank 2 type to handle arbitrary numeric data constructor
handleNumericBinOp
  :: (forall a. (Numeric a) => Expr a -> Expr a -> BinOp a)
  -> P.ParseExpr
  -> P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
handleNumericBinOp ctr e1 e2 = do
  e <- handleNumericExpr e1 
  case e of
    Right (IntExpr ie1) -> do
      x <- handleNumericExpr e2
      case x of 
        Right (IntExpr ie2) -> pure $ Right $ IntExpr $ BinOp $ ctr ie1 ie2
        Right _ -> pure $ Left $ TypeError $ "Argument mismatch" 
        Left x -> pure $ Left x
    Right (RealExpr re1) -> do 
      x <- handleNumericExpr e2
      case x of
        Right (RealExpr re2) -> pure $ Right $ RealExpr $ BinOp $ ctr re1 re2
        Right _ -> pure $ Left $ TypeError $ "Argument mismatch" 
        Left x -> pure $ Left x
    Left x -> pure $ Left x

handleNumericExpr
  :: P.ParseExpr
  -> Reader Env (Either StaticError NumericExpr)
handleNumericExpr expr = case expr of
  P.ParseInt i -> pure $ Right $ IntExpr $ Lit $ IntLit i
  P.ParseReal r -> pure $ Right $ RealExpr $ Lit $ RealLit r
  P.ParseRef r -> undefined
  P.ParseNegate n -> handleNumericUnOp Negate n
  P.ParseTimes e1 e2 -> handleNumericBinOp Times e1 e2 
  P.ParsePlus e1 e2 -> handleNumericBinOp Plus e1 e2 
  other -> pure $ Left $ TypeError $ "Expecting numeric, got " 
                              <> show other

handleBoolExpr 
  :: P.ParseExpr 
  -> Reader Env (Either StaticError (Expr BoolType))
handleBoolExpr expr = case expr of
  P.ParseBool b -> pure $ Right $ Lit $ BoolLit b
  P.ParseRef r -> undefined
  other -> pure $ Left $ TypeError $ "Expecting boolean, got " 
                                     <> show other


handleExpr
  :: P.ParseExpr 
  -> Reader Env (Either StaticError GenericExpr)
handleExpr expr = case expr of
  P.ParseBool b -> pure $ Right $ GenericExpr $ Lit $ BoolLit b
  P.ParseInt i -> pure $ Right $ GenericExpr $ Lit $ IntLit i
  P.ParseReal r -> pure $ Right $ GenericExpr $ Lit $ RealLit r
  P.ParseRef pr -> do 
    env <- ask
    case pr of
      P.Var name ->  
        case Map.lookup (Name name) (env^.objects) of
          Just _ -> undefined
          Nothing -> undefined
      P.Port name port -> 
        case Map.lookup (Name name) (env^.objects) of
          Nothing -> undefined
          Just _ -> undefined
  P.ParseNegate n -> do
    e <- handleNumericUnOp Negate n
    case e of
      Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
      Right (RealExpr re) -> pure $ Right $ GenericExpr re
      Left x -> pure $ Left x
  P.ParsePlus e1 e2 -> do
    e <- handleNumericBinOp Plus e1 e2 
    case e of
      Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
      Right (RealExpr re) -> pure $ Right $ GenericExpr re
      Left x -> pure $ Left x
  P.ParseTimes e1 e2 -> do
    e <- handleNumericBinOp Times e1 e2 
    case e of
      Right (IntExpr ie) -> pure $ Right $ GenericExpr ie
      Right (RealExpr re) -> pure $ Right $ GenericExpr re
      Left x -> pure $ Left x

handleCreate = undefined

handleShow
  :: P.Ref 
  -> Reader Env (Either StaticError String)
handleShow ref = do
  env <- ask
  case ref of
    P.Var x -> do
      obj <- getObj (Name x)
      pure $ Right $ show obj
    P.Port x p -> do
      undefined 
      {--
      case getObj (Name x) of 
        Expr _ _ _-> pure $ Left $ Invalid x
        Module m _-> case getPort p m of
          Nothing -> pure $ Left $ Unspecified p
          Just (Input x) -> pure $ Right $ show x
          Just (Output x) -> pure $ Right $ show x
      --}
handleParse 
  :: P.ParseResult 
  -> State Runtime (Maybe String)
handleParse result = do
  rt <- get
  let e = rt^.env
  case result of 
    P.Show ref -> do 
      let result = runReader (handleShow ref) e
      case result of
        Left err -> pure $ Just $ show err
        Right expr -> pure $ Just expr
    P.Listen x -> do
      buildListen x
      pure Nothing
    P.Assign ref expr -> do 
      if nameExists ref e
        then pure $ Just $ show ref <> " already declared"
        else buildAssign ref expr >> pure Nothing
    P.Create ref obj -> do 
      if nameExists ref e
        then pure $ Just $ show ref <> " already declared"
        else buildObj ref obj >> pure Nothing

generateCExpr :: Env -> Expr a -> IO (Ptr CExpr)
generateCExpr = undefined