module Parser where

import Expression
import Object
import Text.Parsec
import Data.Foldable
import Control.Monad
import Foreign.Ptr
import Control.Lens

import qualified Data.Map as Map

data Ref 
  = ObjRef String String 
  | VarRef String
  deriving (Show)

data ParseTree = Assign Ref Expr
  deriving (Show)


type Parser = Parsec String ()

line :: Parser Expr
line = spaces >> (expr <* eof)

num :: Parser Expr
num = (Literal . read) <$> lexeme (many1 digit)

unOp :: Parser (Expr -> Expr)
unOp = (symbol '-' >> pure Negate)

lexeme = flip (<*) spaces
symbol = lexeme . char
parens = between (symbol '(') (symbol ')')

-- apply 0 to n unary operator constructors to expression 
factor :: Parser Expr
factor = do
  u <- many unOp
  i <- num <|> parens expr
  pure $ foldr ($) i u

-- parse 1 to n terms on lhs of '*'
term :: Parser Expr
term = chainl1 factor $ symbol '*' >> pure Times

expr :: Parser Expr
expr = chainl1 term $ symbol '+' >> pure Plus

-- parse reference of the form "object.property"
parseObjRef :: Parser (String, String)
parseObjRef = do
  -- don't consume trailing whitespace!
  s1 <- many1 letter
  _ <- char '.'
  s2 <- lexeme $ many1 letter
  pure $ (s1, s2)

-- parse reference of the form "variable"
parseVarRef :: Parser String
parseVarRef = lexeme $ many1 letter

assign :: Parser Var
assign = do
  v <- lexeme $ many1 letter
  _ <- lexeme $ char '='
  e <- lexeme expr <* eof
  pure $ Var
    { _name = Name v
    , _this = nullPtr
    , _obj = Expression e
    }

parseVCO :: Parser Var
parseVCO = do
  _ <- try $ string "vco"
  _ <- space >> spaces
  varName <- (lexeme $ many1 letter) <* eof
  pure $ Var 
    { _name = Name varName
    , _this = nullPtr
    , _obj = VCO (Freq 1.0) (FM $ Literal 1.0)
    } 


parseVar :: Parser Var
parseVar = do
  parseVCO