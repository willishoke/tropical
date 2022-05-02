module Parser where

import Expression
import Object
import Text.Parsec
import Data.Foldable
import Control.Monad
import Foreign.Ptr
import Control.Lens

--import qualified Data.Map as Map

data Command
  = Listen 
  | Assignment
  deriving (Show)

parseRef :: Parser Ref
parseRef = do
  s1 <- many1 letter
  x <- optionMaybe $ char '.' >> many1 letter
  _ <- spaces *> (lexeme $ char '=')
  pure $ case x of
    Nothing -> VarID s1
    Just s2 -> ModuleField s1 s2


-- Assign represents any statement containing '='
-- May not be well-formed and requires verification
-- Examples of well-formed statements:
-- z = 3 + y
-- x.input = 10
-- x = new vco
data Assign 
  = Assign Ref Object
  deriving (Show)


parseAssign :: Parser Assign
parseAssign = do
  lhs <- parseRef
  --m <- parseAssignModule lhs | Expression expr)
  undefined

parseAssignModule :: Ref -> Parser Assign
parseAssignModule lhs = do
  _ <- lexeme1 $ string "new"
  m <- parseModule
  pure $ Assign lhs $ Module m

parseModule :: Parser Module
parseModule = choice
  [ parseVCO
  ] 

parseVCO :: Parser Module
parseVCO = do
  _ <- try $ lexeme1 $ string "vco"
  pure $ VCO (Freq 1.0) (FM $ Literal 0.0)

type Parser = Parsec String ()

line :: Parser Expr
line = spaces >> (expr <* eof)

num :: Parser Expr
num = (Literal . read) <$> lexeme (many1 digit)

unOp :: Parser (Expr -> Expr)
unOp = (symbol "-" >> pure Negate)

lexeme = flip (<*) spaces
lexeme1 = flip (<*) (space >> spaces)
symbol = lexeme . string
parens = between (symbol "(") (symbol ")")

-- apply 0 to n unary operator constructors to expression 
factor :: Parser Expr
factor = do
  u <- many unOp
  i <- num <|> parens expr
  pure $ foldr ($) i u

-- parse 1 to n terms on lhs of '*'
term :: Parser Expr
term = chainl1 factor $ symbol "*" >> pure Times

expr :: Parser Expr
expr = chainl1 term $ symbol "+" >> pure Plus