{-# LANGUAGE DeriveDataTypeable #-}

-- This module converts strings to parse trees
-- Validation and type checking in Analyzer module

-- Should be imported qualified to avoid namespace conflicts

module Parser where

import Control.Applicative (liftA2)
import Control.Lens
import Control.Monad

import Data.Foldable
import Data.Typeable

import Foreign.Ptr

import Text.Parsec

type Signal = Double 

-- Useful combinators for parsing:

-- Consume 1+ whitespace elements 
spaces1 = space >> spaces

-- Consume 0+ trailing whitespace elements
lexeme = flip (<*) spaces

-- Discard 1+ trailing whitespace elements
lexeme1 = flip (<*) spaces1 

-- Match a string, discarding trailing whitespace
symbol = lexeme . string

-- Parse input between parens 
parens = between (symbol "(") (symbol ")")

data Ref
  = Var String
  | Port String String
  deriving (Show) 

data ParseModule
  = VCO 
    { baseFreq :: ParseExpr
    , basePhase :: ParseExpr
    }
  deriving (Show)

data ParseExpr
  = ParseReal Double
  | ParseInt Int
  | ParseBool Bool
  | ParseRef Ref
  | ParseNegate ParseExpr
  | ParseTimes ParseExpr ParseExpr
  | ParsePlus ParseExpr ParseExpr
  deriving (Show)

data ParseResult
  = Show Ref
  | Listen ParseExpr
  | Assign Ref ParseExpr 
  | Create Ref ParseObject
  deriving (Show)

data ParseObject
  = ParseModule ParseModule
  | ParseExpr ParseExpr
  deriving (Show)

data Assignment 
  = Assignment Ref ParseExpr
  deriving (Show)


parseResult :: Parser ParseResult
parseResult = do
  undefined 

ref :: Parser Ref
ref = do
  s1 <- many1 letter
  x <- optionMaybe $ char '.' >> many1 letter
  _ <- spaces1
  pure $ case x of
    Nothing -> Var s1
    Just s2 -> Port s1 s2

newModule :: Ref -> Parser Assignment
newModule lhs = do
  _ <- lexeme1 $ string "new"
  m <- parseModule 
  undefined 

parseModule :: Parser ParseModule
parseModule = choice
  [ vco 
  ] 

vco :: Parser ParseModule
vco = do
  _ <- try $ lexeme1 $ string "vco"
  --pure $ VCO (Freq 1.0) (FM $ Literal 0.0)
  undefined

type Parser = Parsec String ()

line :: Parser ParseExpr
line = spaces >> (expr <* eof)

num :: Parser ParseExpr
num = (ParseReal . read) <$> lexeme (many1 digit)

unOp :: Parser (ParseExpr -> ParseExpr)
unOp = (symbol "-" >> pure ParseNegate)

-- apply 0 to n unary operator constructors to expression 
factor :: Parser ParseExpr
factor = do
  u <- many unOp
  i <- num <|> parens expr
  pure $ foldr ($) i u

-- parse 1 to n terms on lhs of '*'
term :: Parser ParseExpr
term = chainl1 factor $ symbol "*" >> pure ParseTimes

expr :: Parser ParseExpr
expr = chainl1 term $ symbol "+" >> pure ParsePlus