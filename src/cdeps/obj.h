#include "expr.h"

class Object
{
    public:

    Object();
    virtual ~Object();
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

    std::unique_ptr<Expression> expr;
};