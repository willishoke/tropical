#include "runtime.h"

extern "C"
{
  void computeC(Runtime* r)
  {
    r->compute();
  }
}

extern "C"
{
  Runtime* initRuntime
  (const unsigned int bufferLength)
  {
    return new Runtime(bufferLength);
  }
}

extern "C"
{
  void deleteRuntime(Runtime* r)
  {
    delete r;
  }
}

extern "C"
{
  void addObject(Runtime* r, Object* obj)
  {
    r->addObject(obj);
  }
}

extern "C"
{
  float* getBufferAddress(Runtime* r)
  {
    return r->rawBuffer;
  }
}