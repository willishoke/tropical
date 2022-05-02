#include <set>
#include <cstdint>

#include "obj.h"

class Runtime
{
  public:

  Runtime
  (const unsigned int bufferLength)
    : bufferLength(bufferLength),
      output(std::make_unique<Output>(bufferLength))
    {
      rawBuffer = output->getBufferAddress();
    }

  void compute()
  {
    for (auto i = 0; i < bufferLength; ++i)
    {
      for (auto& obj : objects) 
      {
        obj->process();
      }
    }
  }

  void addObject(Object* obj)
  {
    objects.insert(std::unique_ptr<Object>(obj));
  }

  float* rawBuffer;

  private:

  std::set<std::unique_ptr<Object>> objects;
  std::unique_ptr<Output> output;
  const unsigned int bufferLength;
};