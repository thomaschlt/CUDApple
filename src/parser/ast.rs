#[derive(Debug, Clone, PartialEq)]
pub enum HostStatement {
    MemoryAllocation {
        variable: String,
        size: Expression,
    },
    MemoryCopy {
        dst: String,
        src: String,
        size: Expression,
        direction: MemcpyKind,
    },
    KernelLaunch {
        kernel: String,
        grid_dim: (Expression, Expression, Expression),
        block_dim: (Expression, Expression, Expression),
        arguments: Vec<Expression>,
    },
    MemoryFree {
        variable: String,
    },
    VariableDeclaration {
        var_type: Type,
        name: String,
    },
    Assignment {
        variable: String,
        value: Expression,
    },
    DeviceSynchronize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug)]
pub struct CudaProgram {
    pub kernels: Vec<KernelFunction>,
    pub host_statements: Vec<HostStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelFunction {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Void,
    Int,
    Float,
    Pointer(Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Declaration(Declaration),
    Assignment(Assignment),
    If(Expression, Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Declaration {
    pub var_type: Type,
    pub name: String,
    pub initializer: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Expression,
    pub value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Variable(String),
    IntegerLiteral(i64),
    SizeOf(String),
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Multiply,
    Add,
    LessThan,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    X,
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dimension::X => write!(f, "x"),
        }
    }
}

use std::fmt;

impl fmt::Display for CudaProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n=== CUDA Program AST ===")?;
        for kernel in &self.kernels {
            write!(f, "{}", kernel)?;
        }
        write!(f, "\n=== End AST ===\n")
    }
}

impl fmt::Display for KernelFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n  Kernel Function: {}", self.name)?;
        writeln!(f, "  Parameters:")?;
        for param in &self.parameters {
            writeln!(f, "    ├─ {}", param)?;
        }
        writeln!(f, "  Body:")?;
        write!(f, "{}", self.body)
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.param_type)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Pointer(t) => write!(f, "{}*", t),
        }
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "    Block {{")?;
        for stmt in &self.statements {
            writeln!(f, "      {}", stmt)?;
        }
        write!(f, "    }}")
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Declaration(d) => write!(f, "{}", d),
            Statement::Assignment(a) => write!(f, "{}", a),
            Statement::If(cond, block) => {
                writeln!(f, "if ({}) {{", cond)?;
                write!(f, "{}", block)?;
                write!(f, "}}")
            }
        }
    }
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.initializer {
            Some(expr) => write!(f, "{} {} = {}", self.var_type, self.name, expr),
            None => write!(f, "{} {}", self.var_type, self.name),
        }
    }
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.target, self.value)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::IntegerLiteral(i) => write!(f, "{}", i),
            Expression::SizeOf(name) => write!(f, "sizeof({})", name),
            Expression::BinaryOp(left, op, right) => write!(f, "({} {} {})", left, op, right),
            Expression::ThreadIdx(d) => write!(f, "ThreadIdx({})", d),
            Expression::BlockIdx(d) => write!(f, "BlockIdx({})", d),
            Expression::BlockDim(d) => write!(f, "BlockDim({})", d),
            Expression::ArrayAccess { array, index } => write!(f, "{}[{}]", array, index),
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Multiply => write!(f, "*"),
            Operator::Add => write!(f, "+"),
            Operator::LessThan => write!(f, "<"),
        }
    }
}
