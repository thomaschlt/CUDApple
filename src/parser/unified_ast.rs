use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct CudaProgram {
    pub host_code: HostCode,
    pub device_code: Vec<KernelFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HostCode {
    pub statements: Vec<HostStatement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Include(String),
    Empty,
    VariableDecl(Declaration),
    Assign(Assignment),
    IfStmt {
        condition: Expression,
        body: Block,
    },
    ForLoop {
        init: Box<Statement>,
        condition: Expression,
        increment: Box<Statement>,
        body: Block,
    },
    CompoundAssign {
        target: Expression,
        operator: Operator,
        value: Expression,
    },
    MacroCall {
        name: String,
        arguments: Vec<Expression>,
    },
    MacroDefinition(MacroDefinition),
}

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
    MultiDeclaration {
        var_type: Type,
        names: Vec<String>,
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
    Declaration(Declaration),
    Assignment(Assignment),
    DeviceSynchronize,
    EventCreate {
        event: String,
    },
    EventRecord {
        event: String,
    },
    EventSynchronize {
        event: String,
    },
    EventElapsedTime {
        milliseconds: String,
        start: String,
        end: String,
    },
    EventDestroy {
        event: String,
    },
    MacroCall {
        name: String,
        arguments: Vec<Expression>,
    },
    MacroDefinition(MacroDefinition),
    FloatDeclaration {
        name: String,
        value: f32,
    },
    CompoundAssignment {
        target: String,
        operator: Operator,
        value: Expression,
    },
    PrintStatement {
        format: String,
        arguments: Vec<Expression>,
    },
    FunctionDecl {
        name: String,
        return_type: Type,
        parameters: Vec<Parameter>,
        body: Block,
    },
    Dim3Declaration {
        name: String,
        x: Expression,
        y: Option<Expression>,
        z: Option<Expression>,
    },
    CudaEventDecl {
        names: Vec<String>,
    },
    CudaApiCall {
        function: String,
        arguments: Vec<Expression>,
    },
    Printf {
        format: String,
        arguments: Vec<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceStatement {
    Declaration(Declaration),
    Assignment(Assignment),
    If(Expression, Block),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Statement>,
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
    Number(i64),
    Variable(String),
    IntegerLiteral(i64),
    SizeOf(Type),
    BinaryOp(Box<Expression>, Operator, Box<Expression>),
    ThreadIdx(Dimension),
    BlockIdx(Dimension),
    BlockDim(Dimension),
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    FunctionCall(String, Vec<Expression>),
    FloatLiteral(f32),
    Constant(String),
    UnaryOp(UnaryOperator, Box<Expression>),
    MemberAccess(Box<Expression>, String),
    AddressOf(Box<Expression>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Void,
    SizeT,
    CudaEventT,
    Dim3,
    Pointer(Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Void => write!(f, "void"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::SizeT => write!(f, "size_t"),
            Type::Dim3 => write!(f, "dim3"),
            Type::CudaEventT => write!(f, "cudaEvent_t"),
            Type::Pointer(inner) => write!(f, "{}*", inner),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    LessThan,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
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
    pub qualifier: Qualifier,
}

#[derive(Debug)]
pub enum ParserError {
    HostCodeError(String),
    DeviceCodeError(String),
    BoundaryError(String),
}

impl std::error::Error for ParserError {}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::HostCodeError(msg) => write!(f, "Host code error: {}", msg),
            ParserError::DeviceCodeError(msg) => write!(f, "Device code error: {}", msg),
            ParserError::BoundaryError(msg) => write!(f, "Program boundary error: {}", msg),
        }
    }
}

impl From<peg::error::ParseError<peg::str::LineCol>> for ParserError {
    fn from(err: peg::error::ParseError<peg::str::LineCol>) -> Self {
        ParserError::HostCodeError(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Qualifier {
    Restrict,
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MacroDefinition {
    FunctionLike {
        name: String,
        parameters: Vec<String>,
        body: String,
    },
    ObjectLike {
        name: String,
        value: String,
    },
}
