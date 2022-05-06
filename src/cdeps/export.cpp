
////////////
// EXPORT //
////////////

// Wrappers for FFI exports
// All calls have side effects, use with care! 

#include "runtime.h"

/*******************/
/* Runtime exports */
/*******************/

// Construct heap-allocated Runtime object
extern "C"
{
  Runtime* initRuntime
  (const unsigned int bufferLength)
  {
    return new Runtime(bufferLength);
  }
}

// Destroy heap-allocated Runtime object
extern "C"
{
  void deleteRuntime(Runtime* r)
  {
    delete r;
  }
}

// Update states sequentially 
// Fill entire output buffer
extern "C"
{
  void computeC(Runtime* r)
  {
    r->compute();
  }
}

// Create new expression
extern "C"
{
  void listen(Runtime* r, Expression* out)
  {
    r->listen(out);
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
    return r->getBufferAddress();
  }
}