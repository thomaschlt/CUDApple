use crate::parser::unified_ast::*;

peg::parser! {
    pub grammar cuda_parser() for str {
        // Basic whitespace and comment handling
        rule _() = quiet!{([' ' | '\t' | '\n' | '\r'] / comment())*}

        rule comment() = block_comment() / line_comment()
        rule block_comment() = "/*" (!"*/" [_])* "*/"
        rule line_comment() = "//" (!"\n" [_])* ("\n" / ![_])

        // Parse a kernel function
        pub rule kernel_function() -> KernelFunction
            = _ "__global__" _ "void" _ name:identifier() _ "(" _ params:parameter_list()? _ ")" _ body:block() {
                KernelFunction {
                    name,
                    parameters: params.unwrap_or_default(),
                    body
                }
            }

        // Basic identifier rule
        rule identifier() -> String
            = id:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                id.to_string()
            }

        // Placeholder rules - expand these based on your needs
        rule qualifier() -> Qualifier
            = "__restrict__" { Qualifier::Restrict }
            / { Qualifier::None }

        rule parameter() -> Parameter
            = param_type:type_specifier() _ "*" _ "__restrict__" _ name:identifier() {
                Parameter {
                    param_type: Type::Pointer(Box::new(param_type)),
                    name: name.to_string(),
                    qualifier: Qualifier::Restrict
                }
            }
            / param_type:type_specifier() _ "*" _ name:identifier() {
                Parameter {
                    param_type: Type::Pointer(Box::new(param_type)),
                    name: name.to_string(),
                    qualifier: Qualifier::None
                }
            }
            / param_type:type_specifier() _ name:identifier() {
                Parameter {
                    param_type,
                    name: name.to_string(),
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
            = "{" _ statements:statement()* _ "}" {
                Block {
                    statements
                }
            }

        rule variable_declaration() -> Statement
            = var_type:type_specifier() _ ptr:"*"? _ name:identifier() _ init:("=" _ e:expression() { e })? _ ";" {
                Statement::VariableDecl(Declaration {
                    var_type: if ptr.is_some() {
                        Type::Pointer(Box::new(var_type))
                    } else {
                        var_type
                    },
                    name,
                    initializer: init,
                })
            }

        rule type_specifier() -> Type
            = "int" { Type::Int }
            / "float" { Type::Float }
            / "void" { Type::Void }
            / "dim3" { Type::Dim3 }

        rule assignment() -> Statement
            = target:(array_access() / (i:identifier() { Expression::Variable(i) })) _ "=" _ value:expression() _ ";" {
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

        rule array_access() -> Expression
            = array:identifier() _ "[" _ index:expression() _ "]" {
                Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(array)),
                    index: Box::new(index)
                }
            }

        // Add these new rules before for_loop()
        rule statement() -> Statement
            = _ s:(
                variable_declaration() /
                assignment() /
                compound_assignment() /
                if_statement() /
                for_loop() /
                include_statement() /
                macro_definition() /
                macro_call() /
                empty_statement()
            ) _ {
                s
            }
        rule empty_statement() -> Statement = ";" { Statement::Empty }
        rule expression() -> Expression = precedence! {
            x:(@) _ "<" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessThan, Box::new(y)) }
            --
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            x:(@) _ "-" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Subtract, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            x:(@) _ "/" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Divide, Box::new(y)) }
            --
            "-" _ x:@ { Expression::UnaryOp(UnaryOperator::Negate, Box::new(x)) }
            --
            f:float_literal() { f }
            c:constant() { c }
            n:number() { Expression::Number(n) }
            m:math_function() { m }
            t:thread_index() { t }
            a:array_access() { a }
            i:identifier() { Expression::Variable(i) }
            "(" _ e:expression() _ ")" { e }
        }
        rule number() -> i64 = n:$(['0'..='9']+) {
            n.parse().unwrap()
        }

        // Add a special rule for for-loop initialization
        rule for_init() -> Statement
            = var_type:type_specifier() _ name:identifier() _ "=" _ e:expression() {
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer: Some(e),
                })
            }

        // Add a special rule for for-loop increment
        rule for_increment() -> Statement
            = target:identifier() _ "++" {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target.clone()),
                    value: Expression::BinaryOp(
                        Box::new(Expression::Variable(target)),
                        Operator::Add,
                        Box::new(Expression::Number(1))
                    )
                })
            }
            / target:identifier() _ "=" _ value:expression() {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target),
                    value
                })
            }

        rule for_loop() -> Statement
            = "for" _ "(" _
              init:for_init() _ ";" _
              condition:expression() _ ";" _
              increment:for_increment() _ ")" _
              body:block() {
                Statement::ForLoop {
                    init: Box::new(init),
                    condition,
                    increment: Box::new(increment),
                    body
                }
            }

        rule float_literal() -> Expression
            = n:$("-"? ['0'..='9']+ "." ['0'..='9']+ "f") {
                Expression::FloatLiteral(n.trim_end_matches('f').parse().unwrap())
            }
            / n:$(['0'..='9']+ "." ['0'..='9']+ "f") {
                Expression::FloatLiteral(n.trim_end_matches('f').parse().unwrap())
            }
            / "1.0f" { Expression::FloatLiteral(1.0) }
            / "0.0f" { Expression::FloatLiteral(0.0) }
            / "-1.0f" { Expression::FloatLiteral(-1.0) }

        rule constant() -> Expression
            = "INFINITY" { Expression::Constant("INFINITY".to_string()) }

        // Add this rule before math_function()
        rule comma_list() -> Vec<Expression>
            = first:expression() rest:(_ "," _ e:expression() { e })* {
                let mut args = vec![first];
                args.extend(rest);
                args
            }

        rule math_function() -> Expression
            = name:$("expf" / "max") _ "(" _ args:comma_list() _ ")" {
                Expression::FunctionCall(name.to_string(), args)
            }

        rule unary_op() -> Expression
            = "-" _ e:expression() {
                Expression::UnaryOp(UnaryOperator::Negate, Box::new(e))
            }

        rule compound_assignment() -> Statement
            = target:identifier() _ op:$("+=" / "-=" / "*=" / "/=") _ value:expression() {
                Statement::CompoundAssign {
                    target: Expression::Variable(target),
                    operator: match op {
                        "+=" => Operator::Add,
                        "-=" => Operator::Subtract,
                        "*=" => Operator::Multiply,
                        "/=" => Operator::Divide,
                        _ => unreachable!()
                    },
                    value
                }
            }

        // Add macro definition rules
        rule macro_definition() -> Statement
            = "#define" _ name:identifier() _ "(" _ params:macro_params() _ ")" _ body:macro_body() {
                Statement::MacroDefinition(MacroDefinition::FunctionLike {
                    name,
                    parameters: params,
                    body: body.trim().to_string(),
                })
            }
            / "#define" _ name:identifier() _ value:macro_body() {
                Statement::MacroDefinition(MacroDefinition::ObjectLike {
                    name,
                    value: value.trim().to_string(),
                })
            }

        rule macro_params() -> Vec<String>
            = first:identifier() rest:(_ "," _ p:identifier() { p })* {
                let mut params = vec![first];
                params.extend(rest);
                params
            }

        rule macro_body() -> String
            = body:$([^'\n']*) _ { body.to_string() }

        // Add macro call rule
        rule macro_call() -> Statement
            = name:identifier() _ "(" _ args:macro_args() _ ")" {
                Statement::MacroCall {
                    name,
                    arguments: args,
                }
            }

        rule macro_args() -> Vec<Expression>
            = args:comma_list()? {
                args.unwrap_or_default()
            }

        // Add this before the existing grammar rules
        rule include_statement() -> Statement
            = "#include" _ "<" path:$([^'>']* ) ">" {
                Statement::Include(path.to_string())
            }
            / "#include" _ "\"" path:$([^'"']* ) "\"" {
                Statement::Include(path.to_string())
            }

        // Add this before the expression() rule
        rule cuda_builtin() -> Expression
            = "CEIL_DIV" _ "(" _ a:expression() _ "," _ b:expression() _ ")" {
                Expression::FunctionCall("CEIL_DIV".to_string(), vec![a, b])
            }
            / "INFINITY" { Expression::Constant("INFINITY".to_string()) }

        rule thread_index() -> Expression
            = "threadIdx" "." "x" { Expression::ThreadIdx(Dimension::X) }
            / "threadIdx" "." "y" { Expression::ThreadIdx(Dimension::Y) }
            / "threadIdx" "." "z" { Expression::ThreadIdx(Dimension::Z) }
            / "blockIdx" "." "x" { Expression::BlockIdx(Dimension::X) }
            / "blockIdx" "." "y" { Expression::BlockIdx(Dimension::Y) }
            / "blockIdx" "." "z" { Expression::BlockIdx(Dimension::Z) }
            / "blockDim" "." "x" { Expression::BlockDim(Dimension::X) }
            / "blockDim" "." "y" { Expression::BlockDim(Dimension::Y) }
            / "blockDim" "." "z" { Expression::BlockDim(Dimension::Z) }
    }
}
