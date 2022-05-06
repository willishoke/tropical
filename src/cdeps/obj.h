#pragma once

#include <vector>

#include "expr.h"

// Would be nice to make this a template class
// parameterized by inputs + outputs, but this
// breaks dynamic binding
class Object
{
  public:

  Object() {}
  virtual ~Object() {}
  virtual void process() = 0;
  virtual void update() = 0;

  protected:

  std::vector<double> inputs;
  std::vector<double> outputs;
};

class VCO : public Object
{
  public:

  VCO(double f, double p);

  void process() override;
  void update() override;

  enum Inputs 
  {
    BASE_FREQ, 
    BASE_PHASE, 
    FM, 
    PM
  };

  enum Outputs
  {
    SIN,
    SQR,
    SAW,
    TRI
  };

  std::unique_ptr<Expression> fm, pm;
  double base_freq, base_phase;
  double sin, sqr, saw, tri;
};


// "everything is an object"
// only top-level expression needs to store value
class Variable : public Object
{
public:
  Variable(Expression* e) : value(0.0)
  {
    expr = std::unique_ptr<Expression>(e);
  }

  ~Variable() {}

  void process() override;
  void update() override;

  double value;

private:
  std::unique_ptr<Expression> expr;
};