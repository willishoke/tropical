module Expression where

import Foreign.C.Types
import Foreign.Ptr

import qualified Data.Set as Set

type Signal = CDouble


data LHS 
    = VarRef String 
    | ObjRef String String

data Expr
    = Literal Double
    | VarRef String
    | ObjRef String String
    | Negate Expr
    | Times Expr Expr
    | Plus Expr Expr
    deriving (Show)

data Assign = Assign 
    { lhs :: Var 
    , rhs :: Var
    }
{--
statement examples:
2      x = 3 + 4 * y  variable = expression
3       x = vco2.sin  variable = object.output
5        vco1.fm = y  object.input = expression
6 vco1.fm = vco2.sin  object.input = object.output
--}


type Env = Set Variable
generateAST :: Set Variable -> Expr -> IO (Ptr CExpr)
generateAST env expr = case expr of
    Literal s -> makeLiteral s
    Ref r -> makeExternal r
    Negate e -> do
        x <- generateAST e
        makeNegate x
    Times e1 e2 -> do
        x <- generateAST e1
        y <- generateAST e2
        makeTimes x y
    Plus e1 e2 -> do
        x <- generateAST e1
        y <- generateAST e2
        makePlus x y

sampleAST :: Expr
sampleAST = Times (Plus (Literal 3) (Literal 2)) (Literal 2)