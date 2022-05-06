#include "runtime.h"

// Assumes ownership of object obj
void Runtime::addObject(Object* obj)
{
  objects.insert(std::unique_ptr<Object>(obj));
}

// Dangerous! Only for use by callback function
float* Runtime::getBufferAddress() 
{
  return buffer.data();
}

// Internal representation uses double for
// precision, but output buffer expects float
void Runtime::fillBuffer(const unsigned int index)
{
  float x = static_cast<float>(output->eval());

  // Interleaved output
  buffer[index] = x;
  buffer[index+1] = x;
}

// Perform a single cycle of computation to
// fill the output buffer
void Runtime::compute()
{
  for (auto i = 0; i < buffer.size(); i += 2)
  {
    // Update state of all objects
    for (auto& obj : objects) 
    {
      obj->process();
    }

    // Update object buffer values
    for (auto& obj : objects) 
    {
      obj->update();
    }

    // Fill output buffer
    fillBuffer(i);
  }
}






void Runtime::listen(Expression* out)
{
  output = std::unique_ptr<Expression>(out);
}

