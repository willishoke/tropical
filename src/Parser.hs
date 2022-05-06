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

--import Expression
import Object

type Signal = Double 

data Ref
  = VarID String
  | ModuleField String String
  deriving (Show) 

data ParseExpr
  = Literal Signal
  | Ref Ref
  | Negate ParseExpr
  | Times ParseExpr ParseExpr
  | Plus ParseExpr ParseExpr
  deriving (Show)

data ParseResult
  = Show ParseExpr
  | Listen ParseExpr
  | Assign Assignment
  deriving (Show)

-- Assignment represents any statement containing '='
-- May not be well-formed and requires verification
-- Examples of well-formed statements:
-- z = 3 + y
-- x.input = 10
-- x = new vco

data Assignment 
  = Assignment Ref ParseExpr
  deriving (Show)


ref :: Parser Ref
ref = do
  s1 <- many1 letter
  x <- optionMaybe $ char '.' >> many1 letter
  _ <- spaces *> (lexeme $ char '=')
  pure $ case x of
    Nothing -> VarID s1
    Just s2 -> ModuleField s1 s2

object :: Parser Object
object = choice
  [  
  ]

--assignment :: Parser Assignment
--assignment = liftA2 Assignment ref object 

newModule :: Ref -> Parser Assignment
newModule lhs = do
  _ <- lexeme1 $ string "new"
  m <- anyModule 
  undefined 

anyModule :: Parser Module
anyModule = choice
  [ vco 
  ] 

vco :: Parser Module
vco = do
  _ <- try $ lexeme1 $ string "vco"
  --pure $ VCO (Freq 1.0) (FM $ Literal 0.0)
  undefined

type Parser = Parsec String ()

line :: Parser ParseExpr
line = spaces >> (expr <* eof)

num :: Parser ParseExpr
num = (Literal . read) <$> lexeme (many1 digit)

unOp :: Parser (ParseExpr -> ParseExpr)
unOp = (symbol "-" >> pure Negate)

lexeme = flip (<*) spaces
lexeme1 = flip (<*) (space >> spaces)
symbol = lexeme . string
parens = between (symbol "(") (symbol ")")

-- apply 0 to n unary operator constructors to expression 
factor :: Parser ParseExpr
factor = do
  u <- many unOp
  i <- num <|> parens expr
  pure $ foldr ($) i u

-- parse 1 to n terms on lhs of '*'
term :: Parser ParseExpr
term = chainl1 factor $ symbol "*" >> pure Times

expr :: Parser ParseExpr
expr = chainl1 term $ symbol "+" >> pure Plus