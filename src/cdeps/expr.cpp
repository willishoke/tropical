#include <string>
#include <vector>
#include <iostream>
#include <thread>

#include "expr.h"

extern "C" 
{
  Signal eval(Expression* expr)
  {
    return expr->eval();
  }
}

extern "C" 
{
  Expression* makeLiteral(Signal s)
  {
    return new Literal(s);
  }
} 

Literal::Literal(Signal s) : value(s) {}

Signal Literal::eval()
{
  return value;
}

External::External(Signal *s) : ref(s) {}

extern "C" 
{
  Expression *makeExternal(Signal *s) 
  {
    return new External(s);
  }
}

Signal External::eval()
{
  return *ref;
}

UnaryOperator::UnaryOperator(Expression* e)
{
  expr = std::unique_ptr<Expression>(e);
}

Negate::Negate(Expression* e) : 
  UnaryOperator(e) {}

Signal Negate::eval() 
{
  return -expr->eval();
}

extern "C"
{
  Expression *makeNegate(Expression* e)
  {
    return new Negate(e);
  }
}

BinaryOperator::BinaryOperator(Expression* e1, Expression* e2) 
{
  first = std::unique_ptr<Expression>(e1);
  second = std::unique_ptr<Expression>(e2);
}

Plus::Plus(Expression* e1, Expression* e2) : 
  BinaryOperator(e1, e2) {}


extern "C" 
{
  Expression* makePlus(Expression* e1, Expression* e2) 
  {
    return new Plus(e1, e2);
  }
}

Signal Plus::eval()
{
  return first->eval() + second->eval();
}

Times::Times(Expression* e1, Expression* e2) : 
  BinaryOperator(e1, e2) {}

extern "C" 
{
  Expression* makeTimes(Expression* e1, Expression* e2) 
  {
    return new Times(e1, e2);
  }
}

Signal Times::eval()
{
  return first->eval() * second->eval();
}