-- This module converts strings to parse trees
-- Validation and type checking in Analyzer module

-- Should be imported qualified to avoid namespace conflicts

module Parser where

import Control.Applicative (liftA2)
import Control.Lens
import Control.Monad

import Data.Foldable

import Foreign.Ptr

import Text.Parsec

type Parser = Parsec String ()

-- COMBINATORS


-- Consume 0+ trailing whitespace elements
lexeme = flip (<*) spaces

-- Match a string, discarding trailing whitespace
symbol = lexeme . string

-- Parse input between parens 
parens = between (symbol "(") (symbol ")")

newtype Ref = Ref { unRef :: String }
  deriving (Show)

data ParseExpr
  = ParseReal Double
  | ParseInt Int
  | ParseBool Bool
  | ParseRef String
  | ParseNegate ParseExpr
  | ParseTimes ParseExpr ParseExpr
  | ParsePlus ParseExpr ParseExpr
  deriving (Show)

data ParseResult
  = Show ParseExpr
  | Listen ParseExpr
  | Assign Ref ParseExpr 
  deriving (Show)

parseResult :: Parser ParseResult
parseResult = do
  undefined 

ref :: Parser Ref
ref = many1 letter >>= pure . Ref

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