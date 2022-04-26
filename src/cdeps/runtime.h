#include <vector>
#include <cstdint>

class Runtime
{
private:
    std::vector<Object> objects;
    std::vector<Expression> exprs;
};