{-# LANGUAGE TemplateHaskell #-}

module Object where

newtype Freq = Freq { freq :: Double }
newtype FM = FM { fm :: Expression}
data Object = VCO Freq FM | Expr Expression

newtype Name = Name { name :: String }
data ObjPtr
data Var = Var 
    { _name :: Name
    , _this :: Ptr ObjPtr
    , _obj :: Object
    }

makeLenses ''Var

instance Eq Var where
    v1 == v2 = v1^.name == v2^.name

instance Ord Var where
    v1 <= v2 = v1^.name <= v2^.name