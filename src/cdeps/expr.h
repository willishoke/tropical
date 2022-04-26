#include <memory>

using Signal = double;

class Expression
{
    public:
    virtual Signal eval() = 0;
    virtual ~Expression() {}
    private:
    //std::string id;
};

class Variable
{
    ~Variable() {}
    private:
    std::unique_ptr<Expression> expr;
    int id;
    Signal value;
};

class Literal : public Expression
{
    public:
    ~Literal() {}
    Literal(Signal);
    Signal eval() override;

    private:
    Signal value;
};

class External : public Expression
{
    public:
    ~External() {}
    External(Signal*);
    Signal eval() override;

    private:
    Signal *ref;
};

class UnaryOperator : public Expression
{
    public:
    virtual ~UnaryOperator() = default;

    protected:
    UnaryOperator(Expression*);
    std::unique_ptr<Expression> expr;

};

class Negate : public UnaryOperator
{
    public:
    Negate(Expression*);
    virtual ~Negate() = default;
    Signal eval() override;
};

class BinaryOperator : public Expression
{
    public:
    virtual ~BinaryOperator() = default;
    protected:
    BinaryOperator(Expression*, Expression*);
    std::unique_ptr<Expression> first;
    std::unique_ptr<Expression> second;
};

class Plus : public BinaryOperator
{
    public:
    virtual ~Plus() = default;
    Plus(Expression*, Expression*);
    Signal eval();
};

class Times : public BinaryOperator
{
    public:
    virtual ~Times() = default;
    Times(Expression*, Expression*); 
    Signal eval();
};