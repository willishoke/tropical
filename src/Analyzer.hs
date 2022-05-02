module Analyzer where

import qualified Expression as E

import Object
import Parser
import Runtime

import Foreign.C.Types

data Literal 
  = BoolLit CBool
  | IntLit  CInt
  | DoubleLit CDouble


data AST
  = Literal Literal