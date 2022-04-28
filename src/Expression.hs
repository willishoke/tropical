module Expression where

import Foreign.C.Types
import Foreign.Ptr

import qualified Data.Set as Set

type Signal = CDouble

data Expr
    = Literal Signal
    | VarRef String
    | ObjRef String String
    | Negate Expr
    | Times Expr Expr
    | Plus Expr Expr
    deriving (Show)

{--
statement examples:
2      x = 3 + 4 * y  variable = expression
3       x = vco2.sin  variable = object.output
5        vco1.fm = y  object.input = expression
6 vco1.fm = vco2.sin  object.input = object.output
--}

