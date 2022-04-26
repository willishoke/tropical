module Parser where

import Expression
import Object
import Text.Parsec
import Data.Foldable
import Control.Monad
import qualified Data.Map as Map

type Parser = Parsec String ()

data ParseExpr 
  = Product Concrete Concrete

eval :: Expr -> Signal
eval e = case e of
  Negate x -> -(eval x)
  Ref x -> undefined
  Times x y -> eval x * eval y
  Plus x y -> eval x + eval y
  Literal x -> x

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
parseObjRef :: Parser ObjRef
objRef = do
  -- don't consume trailing whitespace!
  s1 <- many1 letter
  _ <- char '.'
  s2 <- lexeme $ many1 letter
  pure $ ObjRef s1 s2

-- parse reference of the form "variable"
parseVarRef :: Parser VarRef
varRef = do
  i <- lexeme $ many1 letter
  pure $ VarRef i

parseRef :: Parser 

parseStmt :: Parser Stmt
stmt = do
  i <- 
  _ <- symbol '='
  e <- expr
  pure $ Assign i e

parseVCO :: Parser Variable
parseVCO = do
  _ <- string "vco"
  _ <- space >> spaces
  varName <- (lexeme $ many1 letter) <* eof
  pure $ Var 
    { name = varName
    , ptr = nullPtr
    , obj = VCO 
      { freq = Literal 1.0
      , fm = Literal 1.0 
      }
    } 


parseObj :: Parser Object
parseObj = do
  objType <- lexeme $ many1 letter
  varName <- lexeme $ many1 letter
  case objType of
    "vco" -> pure $ VCO { freq = 1.0, fm = 1.0 }
