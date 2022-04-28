#include <set>
#include <cstdint>

#include "obj.h"

class Runtime
{
    public:
    std::set<std::unique_ptr<Object>> objects;
    void compute() {};
};