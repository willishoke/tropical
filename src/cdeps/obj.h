#include <vector>

#include "expr.h"

class Object
{
  public:

  Object() {}
  virtual ~Object() {}
  virtual void process() = 0;
};

using Signal = double;

class VCO : public Object
{
  public:

  VCO(Signal f);
  VCO(Signal f, Signal p);
  void process() override;

  std::unique_ptr<Expression> fm, pm;
  Signal base_freq, base_phase;
  Signal sin, sqr, saw, tri;
};

// Intended to be used as a singleton type,
// instantiated once at runtime
class Output : public Object
{
  public:
  Output(unsigned int bufferSize)
    : signal(std::make_unique<Literal>(0.0)),
      buffer(bufferSize),
      bufferSize(bufferSize),
      bufferPosition(0)
  {}

  ~Output() {}

  void process() override 
  {
    // internal representation uses double for improved precision,
    // but output buffer expects float
    float x = static_cast<float>(signal->eval());
    // interleaved output
    buffer[bufferPosition++] = x;
    buffer[bufferPosition++] = x;
    bufferPosition %= bufferSize;
  }

  float* getBufferAddress()
  {
    return buffer.data();
  }

  private:

  // TODO: support for 1..n channels
  std::unique_ptr<Expression> signal;
  std::vector<float> buffer;
  unsigned int bufferSize;
  unsigned int bufferPosition;
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
  void process() override
  {
    value = expr->eval();
  }

  Signal value;

private:
  // indicates whether or not
  bool listen;
  std::unique_ptr<Expression> expr;
};