{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

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

import qualified Data.Map as Map
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
    :: DataType a
    => Expr a 
    -> Ptr (CValue a)
    -> Object

instance Show Object where
  show (Module m _) = show m
  show (Expr e _) = show e

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
  :: Env
  -> Name
  -> Either StaticError Object
getObj e n = case Map.lookup n (e^.objects) of 
  Nothing -> Left $ Undeclared $ unName n
  Just obj -> Right obj


-- Here's where the Data.Typeable magic comes in:
-- use casts to peek at phantom type variables at runtime

resolveRef
  :: Env
  -> P.Ref 
  -> Either StaticError GenericExpr
resolveRef env ref = case ref of
  P.Var name -> 
    getObj env (Name name) >>= \obj -> case obj of
      Module m p -> Left $ Unspecified name
      Expr e p -> undefined
  P.Port objName portName -> 
    getObj env (Name objName) >>= \obj -> case obj of
      Expr _ _ -> Left $ Invalid objName
      Module m _ -> case getPort portName m of
        Nothing -> Left $ Invalid portName 
        Just (Input _) -> Right undefined
        Just (Output _) -> Right undefined

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
  :: Env 
  -> P.Ref
  -> Either StaticError NumericExpr 
handleNumericRef = undefined

handleNumericUnOp
  :: (forall a. (Numeric a) => Expr a -> UnOp a)
  -> Env 
  -> P.ParseExpr
  -> Either StaticError NumericExpr 
handleNumericUnOp ctr env expr = case handleNumericExpr env expr of
  Right (IntExpr ie) -> Right $ IntExpr $ UnOp $ ctr ie
  Right (RealExpr re) -> Right $ RealExpr $ UnOp $ ctr re
  Left x -> Left x

-- Using RankNTypes to handle arbitrary numeric data constructor
handleNumericBinOp
  :: (forall a. (Numeric a) => Expr a -> Expr a -> BinOp a)
  -> Env 
  -> P.ParseExpr
  -> P.ParseExpr
  -> Either StaticError NumericExpr 
handleNumericBinOp ctr env e1 e2 = 
  case handleNumericExpr env e1 of
    Right (IntExpr ie1) -> case handleNumericExpr env e2 of
      Right (IntExpr ie2) -> Right $ IntExpr $ BinOp $ ctr ie1 ie2
      Right _ -> Left $ TypeError $ "Argument mismatch" 
      Left x -> Left x
    Right (RealExpr re1) -> case handleNumericExpr env e2 of
      Right (RealExpr re2) -> Right $ RealExpr $ BinOp $ ctr re1 re2
      Right _ -> Left $ TypeError $ "Argument mismatch" 
      Left x -> Left x
    Left x -> Left x

handleNumericExpr
  :: Env
  -> P.ParseExpr
  -> Either StaticError NumericExpr
handleNumericExpr env expr = case expr of
  P.ParseInt i -> Right $ IntExpr $ Lit $ IntLit i
  P.ParseReal r -> Right $ RealExpr $ Lit $ RealLit r
  P.ParseRef r -> undefined
  P.ParseNegate n -> handleNumericUnOp Negate env n
  P.ParseTimes e1 e2 -> handleNumericBinOp Times env e1 e2 
  P.ParsePlus e1 e2 -> handleNumericBinOp Plus env e1 e2 
  other -> Left $ TypeError $ "Expecting numeric, got " 
                              <> show other

handleBoolExpr 
  :: Env 
  -> P.ParseExpr 
  -> Either StaticError (Expr BoolType)
handleBoolExpr env expr = case expr of
  P.ParseBool b -> Right $ Lit $ BoolLit b
  P.ParseRef r -> undefined
  other -> Left $ TypeError $ "Expecting boolean, got " 
                              <> show other



handleExpr
  :: Env
  -> P.ParseExpr 
  -> Either StaticError GenericExpr
handleExpr env expr = case expr of
  P.ParseBool b -> Right $ GenericExpr $ Lit $ BoolLit b
  P.ParseInt i -> Right $ GenericExpr $ Lit $ IntLit i
  P.ParseReal r -> Right $ GenericExpr $ Lit $ RealLit r
  P.ParseRef pr -> case pr of
    P.Var name -> 
      case Map.lookup (Name name) (env^.objects) of
        Just _ -> undefined
        Nothing -> undefined
    P.Port name port -> 
      case Map.lookup (Name name) (env^.objects) of
        Nothing -> undefined
        Just _ -> undefined
  P.ParseNegate n -> 
    let e = handleNumericUnOp Negate env n
    in case e of
      Right (IntExpr ie) -> Right $ GenericExpr ie
      Right (RealExpr re) -> Right $ GenericExpr re
      Left x -> Left x
  P.ParsePlus e1 e2 -> 
    let e = handleNumericBinOp Plus env e1 e2 
    in case e of
      Right (IntExpr ie) -> Right $ GenericExpr ie
      Right (RealExpr re) -> Right $ GenericExpr re
      Left x -> Left x
  P.ParseTimes e1 e2 -> 
    let e = handleNumericBinOp Times env e1 e2 
    in case e of
      Right (IntExpr ie) -> Right $ GenericExpr ie
      Right (RealExpr re) -> Right $ GenericExpr re
      Left x -> Left x

handleCreate = undefined

handleShow
  :: Env 
  -> P.Ref 
  -> Either StaticError String
handleShow env ref = 
  case ref of
    P.Var x -> getObj env (Name x) >>= (Right . show)
    P.Port x p -> getObj env (Name x) >>= \obj ->
      case obj of 
        Expr _ _ -> Left $ Invalid x
        Module m _-> case getPort p m of
          Nothing -> Left $ Unspecified p
          Just (Input x) -> Right $ show x
          Just (Output x) -> Right $ show x

handleParse 
  :: P.ParseResult 
  -> InputT (StateT Runtime IO) ()
handleParse result = do
  rt <- lift get
  let e = rt^.env
  case result of 
    P.Show ref -> do 
      let result = handleShow e ref
      case result of
        Left x -> outputStrLn $ show x
        Right toShow -> outputStrLn toShow
    P.Eval x -> do
      let repr = handleExpr e x 
      case repr of
        Left err -> outputStrLn $ show err
        Right obj -> outputStrLn $ show obj 
    P.Listen x -> do
      buildListen x
    P.Assign ref expr -> do 
      if nameExists ref e 
        then outputStrLn $ (show ref) <> " already declared"
        else buildAssign ref expr 
    P.Create ref obj -> do 
      if nameExists ref e
        then outputStrLn $ (show ref) <> " already declared"
        else buildObj ref obj

 {--
  TODO: Expr a -> Object

  currentRT <- lift get
  crepr <- liftIO $ generateCRepr (currentRT^.env) result 
  let v = Var 
            { _name = Name ""
            , _this = castPtr crepr
            , _obj = Expr result 
            }
  lift $ put $ over env (Set.insert v) currentRT 
  evaluated <- liftIO $ evalc crepr
  outputStrLn $ show evaluated
--}

generateCExpr :: Env -> Expr a -> IO (Ptr CExpr)
generateCExpr = undefined