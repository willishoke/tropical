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
    Runtime* initRuntime()
    {
        return new Runtime(); 
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
        r->objects.insert(std::unique_ptr<Object>(obj));
    }
}