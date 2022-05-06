#pragma once

#include <set>
#include <cstdint>

#include "obj.h"

// TODO: paramterize output settings
// (channels, sammple rate, oversampling, etc)

class Runtime
{
  public:

  Runtime(const unsigned int bufferSize)
    : output(std::make_unique<Literal>(0.0)),
      buffer(bufferSize)
    {}

  void compute();
  void fillBuffer(const unsigned int index);

  void addObject(Object* obj);


  void fillBuffer();

  float* getBufferAddress();

  // Current implementation broadcasts mono signal 
  // into left and right stereo channels
  // TODO: support for 1..n channels
  void listen(Expression* out);

  private:

  std::set<std::unique_ptr<Object>> objects;
  std::unique_ptr<Expression> output;
  std::vector<float> buffer;
};