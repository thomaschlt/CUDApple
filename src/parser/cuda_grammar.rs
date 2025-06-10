use crate::parser::unified_ast::*;

peg::parser! {
    pub grammar cuda_parser() for str {
        rule _() = quiet!{([' ' | '\t' | '\n' | '\r'] / comment())*}

        rule comment() = block_comment() / line_comment()
        rule block_comment() = "/*" (!"*/" [_])* "*/"
        rule line_comment() = "//" (!"\n" [_])* ("\n" / ![_])

        pub rule cuda_program() -> CudaProgram
            = _ functions:(function_definition() ** _) _ {
                let mut kernels = Vec::new();
                let mut device_functions = Vec::new();

                for func in functions {
                    match func {
                        FunctionDefinition::Kernel(k) => kernels.push(k),
                        FunctionDefinition::Device(d) => device_functions.push(d),
                    }
                }

                CudaProgram {
                    device_code: kernels,
                    device_functions,
                }
            }

        rule function_definition() -> FunctionDefinition
            = k:kernel_function() { FunctionDefinition::Kernel(k) }
            / d:device_function() { FunctionDefinition::Device(d) }

        pub rule kernel_function() -> KernelFunction
            = _ "__global__" _ "void" _ name:identifier() _ "(" _ params:parameter_list()? _ ")" _ body:block() {
                KernelFunction {
                    name,
                    parameters: params.unwrap_or_default(),
                    body
                }
            }

        rule device_function() -> DeviceFunction
            = _ "__device__" _ return_type:type_specifier() _ name:identifier() _ "(" _ params:parameter_list()? _ ")" _ body:block() {
                DeviceFunction {
                    name,
                    return_type,
                    parameters: params.unwrap_or_default(),
                    body
                }
            }

        rule identifier() -> String
            = id:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                id.to_string()
            }

        rule qualifier() -> Qualifier
            = "__restrict__" { Qualifier::Restrict }
            / { Qualifier::None }

        rule parameter() -> Parameter
            = param_type:type_specifier() _ "*" _ qualifier:qualifier() _ name:identifier() {
                Parameter {
                    param_type: Type::Pointer(Box::new(param_type)),
                    name,
                    qualifier
                }
            }
            / param_type:type_specifier() _ name:identifier() {
                Parameter {
                    param_type,
                    name,
                    qualifier: Qualifier::None
                }
            }

        rule parameter_list() -> Vec<Parameter>
            = first:parameter() rest:(_ "," _ p:parameter() { p })* {
                let mut params = vec![first];
                params.extend(rest);
                params
            }

        rule block() -> Block
            = _ "{" _ statements:(statement() ** _) _ "}" _ {
                Block { statements }
            }

        rule statement() -> Statement
            = for_loop()
            / variable_declaration()
            / compound_assignment()
            / assignment()
            / if_statement()
            / return_statement()
            / e:expression() _ ";" { Statement::Expression(Box::new(e)) }

        rule variable_declaration() -> Statement
            = "__shared__" _ var_type:type_specifier() _ name:identifier() _ array_size:array_size()? _ ";" {
                let var_type = if let Some(size) = array_size {
                    Type::Array(Box::new(var_type), size)
                } else {
                    var_type
                };
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer: None,
                    is_shared: true,
                })
            }
            / var_type:type_specifier() _ name:identifier() _ array_size:array_size()? _ initializer:("=" _ expr:expression() { expr })? _ ";" {
                let var_type = if let Some(size) = array_size {
                    Type::Array(Box::new(var_type), size)
                } else {
                    var_type
                };
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer,
                    is_shared: false,
                })
            }

        rule type_specifier() -> Type
            = "int" { Type::Int }
            / "float" { Type::Float }
            / "void" { Type::Void }

        rule assignment() -> Statement
            = target:(array_access() / variable()) _ "=" _ value:expression() _ ";" {
                Statement::Assign(Assignment {
                    target,
                    value
                })
            }

        rule if_statement() -> Statement
            = "if" _ "(" _ condition:expression() _ ")" _ body:block() {
                Statement::IfStmt {
                    condition,
                    body
                }
            }
            / "if" _ "(" _ condition:expression() _ ")" _ stmt:statement() {
                Statement::IfStmt {
                    condition,
                    body: Block { statements: vec![stmt] }
                }
            }

        rule for_loop() -> Statement
            = "for" _ "(" _
              init:for_init() _ ";" _
              condition:expression() _ ";" _
              increment:for_increment() _
              ")" _
              body:block() {
                Statement::ForLoop {
                    init: Box::new(init),
                    condition,
                    increment: Box::new(increment),
                    body,
                }
            }

        rule for_init() -> Statement
            = var_type:type_specifier() _ name:identifier() _ "=" _ value:expression() {
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer: Some(value),
                    is_shared: false,
                })
            }

        rule for_increment() -> Statement
            = target:identifier() _ "++" {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target.clone()),
                    value: Expression::BinaryOp(
                        Box::new(Expression::Variable(target)),
                        Operator::Add,
                        Box::new(Expression::IntegerLiteral(1))
                    )
                })
            }
            / target:identifier() _ "=" _ target2:identifier() _ "+" _ value:expression() {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target),
                    value: Expression::BinaryOp(
                        Box::new(Expression::Variable(target2)),
                        Operator::Add,
                        Box::new(value)
                    )
                })
            }
            / target:identifier() _ "+=" _ value:expression() {
                Statement::CompoundAssign {
                    target: Expression::Variable(target),
                    operator: Operator::Add,
                    value
                }
            }

        rule array_access() -> Expression
            = array:identifier() _ "[" _ index:expression() _ "]" {
                Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(array)),
                    index: Box::new(index)
                }
            }

        rule variable() -> Expression
            = name:identifier() { Expression::Variable(name) }

        rule math_function() -> Expression
            = "fmaxf" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "fmaxf".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "fminf" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "fminf".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "max" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "max".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "min" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "min".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "expf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction {
                    name: "expf".to_string(),
                    arguments: vec![x],
                }
            }
            / "logf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction {
                    name: "logf".to_string(),
                    arguments: vec![x],
                }
            }
            / "sqrtf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction {
                    name: "sqrtf".to_string(),
                    arguments: vec![x],
                }
            }
            / "powf" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "powf".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "tanhf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction {
                    name: "tanhf".to_string(),
                    arguments: vec![x],
                }
            }
            / "fabsf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction {
                    name: "fabsf".to_string(),
                    arguments: vec![x],
                }
            }
            / "atomicAdd" _ "(" _ target:expression() _ "," _ value:expression() _ ")" {
                Expression::AtomicOperation {
                    operation: "atomicAdd".to_string(),
                    target: Box::new(target),
                    value: Box::new(value),
                }
            }
            / name:identifier() _ "(" _ args:argument_list()? _ ")" {
                Expression::FunctionCall {
                    name,
                    arguments: args.unwrap_or_default(),
                }
            }

        rule argument_list() -> Vec<Expression>
            = first:expression() rest:(_ "," _ e:expression() { e })* {
                let mut args = vec![first];
                args.extend(rest);
                args
            }

        rule infinity() -> Expression
            = "INFINITY" { Expression::Infinity }
            / "-INFINITY" { Expression::NegativeInfinity }
            / "-" _ n:number() _ "*" _ "INFINITY" { Expression::NegativeInfinity }
            / n:number() _ "*" _ "-" _ "INFINITY" { Expression::NegativeInfinity }

        rule expression() -> Expression = precedence! {
            x:(@) _ "&&" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LogicalAnd, Box::new(y))}
            x:(@) _ "||" _ y:@ {Expression::BinaryOp(Box::new(x), Operator::LogicalOr, Box::new(y))}
            --
            x:(@) _ "==" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Equal, Box::new(y)) }
            x:(@) _ "!=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::NotEqual, Box::new(y)) }
            x:(@) _ "<=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessEqual, Box::new(y)) }
            x:(@) _ ">=" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::GreaterEqual, Box::new(y)) }
            x:(@) _ "<" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessThan, Box::new(y)) }
            x:(@) _ ">" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::GreaterThan, Box::new(y)) }
            --
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            x:(@) _ "-" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Subtract, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            x:(@) _ "/" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Divide, Box::new(y)) }
            x:(@) _ "%" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Modulo, Box::new(y)) }
            --
            "&" _ e:@ { Expression::UnaryOp(UnaryOperator::AddressOf, Box::new(e)) }
            "*" _ e:@ { Expression::UnaryOp(UnaryOperator::Dereference, Box::new(e)) }
            "-" _ e:@ { Expression::UnaryOp(UnaryOperator::Negate, Box::new(e)) }
            --
            n:number() { n }
            i:infinity() { i }
            t:thread_index() { t }
            m:math_function() { m }
            a:array_access() { a }
            v:variable() { v }
            "(" _ e:expression() _ ")" { e }
        }

        rule number() -> Expression
            // Scientific notation with decimal: 1.23e-4f, 2.5e10f, etc.
            = n:$(['0'..='9']+ "." ['0'..='9']* ['e' | 'E'] ['+' | '-']? ['0'..='9']+ "f"?) {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
            // Scientific notation without decimal: 1e-8f, 2e10f, etc.
            / n:$(['0'..='9']+ ['e' | 'E'] ['+' | '-']? ['0'..='9']+ "f"?) {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
            // Regular decimal: 1.23f, 4.56, etc.
            / n:$(['0'..='9']+ "." ['0'..='9']* "f"?) {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
            // Float with f suffix: 123f
            / n:$(['0'..='9']+ "f") {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
            // Regular integer: 123
            / n:$(['0'..='9']+) {
                Expression::IntegerLiteral(n.parse().unwrap())
            }

        rule thread_index() -> Expression
            = "threadIdx." d:dimension() { Expression::ThreadIdx(d) }
            / "blockIdx." d:dimension() { Expression::BlockIdx(d) }
            / "blockDim." d:dimension() { Expression::BlockDim(d) }

        rule dimension() -> Dimension
            = "x" { Dimension::X }
            / "y" { Dimension::Y }
            / "z" { Dimension::Z }

        rule compound_assignment() -> Statement
            = target:(array_access() / variable()) _ op:compound_operator() _ value:expression() _ ";" {
                Statement::CompoundAssign {
                    target,
                    operator: op,
                    value
                }
            }

        rule compound_operator() -> Operator
            = "+=" { Operator::Add }
            / "-=" { Operator::Subtract }
            / "*=" { Operator::Multiply }
            / "/=" { Operator::Divide }
            / "%=" { Operator::Modulo }

        rule return_statement() -> Statement
            = "return" _ expr:expression() _ ";" {
                Statement::Return(Some(Box::new(expr)))
            }
            / "return" _ ";" {
                Statement::Return(None)
            }

        rule array_size() -> i64
            = "[" _ size:$(['0'..='9']+) _ "]" {
                size.parse().unwrap()
            }
    }
}
