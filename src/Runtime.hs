{-# LANGUAGE TemplateHaskell #-}

module Runtime where

import Control.Lens
import Control.Monad.Reader
import Control.Monad.State

import qualified Data.Map as Map

import Foreign.C.Types
import Foreign.Ptr

import Analyzer
import Interface
import qualified Parser as P 


data Runtime = Runtime
  { _env :: Env
  , _runtime :: Ptr CRuntime
  }
makeLenses ''Runtime

handleParse 
  :: P.ParseResult 
  -> State Runtime (Maybe String)
handleParse result = do
  rt <- get
  let e = rt^.env
  case result of 
    P.Show expr -> do 
      let result = runReader (handleShow expr) e
      case result of
        Left err -> pure $ Just $ show err
        Right expr -> pure $ Just expr
    P.Listen x -> undefined
    P.Assign ref expr -> undefined

handleShow
  :: P.ParseExpr
  -> Reader Env (Either StaticError String)
handleShow expr = case expr of
  P.ParseRef ref -> do
    expr <- getStoredExpr $ Name $ ref
    undefined
  expr -> pure $ Right $ show expr
  
generateCExpr :: Env -> Expr a -> IO (Ptr CExpr)
generateCExpr = undefined