module Expression where

import Foreign.C.Types
import Foreign.Ptr

import qualified Data.Set as Set

type Signal = CDouble

data Ref
  = VarID String
  | ModuleField String String
  deriving (Show) 

data Expr
  = Literal Signal
  | Ref Ref
  | Negate Expr
  | Times Expr Expr
  | Plus Expr Expr
  deriving (Show)

{--
statement examples:
      x = 3 + 4 * y  variable = expression
       x = vco2.sin  variable = object.output
        vco1.fm = y  object.input = expression
 vco1.fm = vco2.sin  object.input = object.output
-}

