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
            = _ s:(
                variable_declaration() /
                assignment() /
                if_statement()
            ) _ {
                s
            }

        rule variable_declaration() -> Statement
            = var_type:type_specifier() _ name:identifier() _ ptr:"*"? _ ";" {
                Statement::VariableDecl(Declaration {
                    var_type: if ptr.is_some() {
                        Type::Pointer(Box::new(var_type))
                    } else {
                        var_type
                    },
                    name,
                    initializer: None,
                })
            }
            / var_type:type_specifier() _ name:identifier() _ init:("=" _ e:expression() { e })? _ ";" {
                Statement::VariableDecl(Declaration {
                    var_type,
                    name,
                    initializer: init,
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
            = "max" _ "(" _ x:expression() _ "," _ y:expression() _ ")" {
                Expression::MathFunction {
                    name: "max".to_string(),
                    arguments: vec![x, y],
                }
            }
            / "expf" _ "(" _ x:expression() _ ")" {
                Expression::MathFunction { name: "expf".to_string(), arguments: vec![x], }
            }

        rule infinity() -> Expression
            = "INFINITY" { Expression::Infinity }
            / "-INFINITY" { Expression::NegativeInfinity }
            / "-" _ n:number() _ "*" _ "INFINITY" { Expression::NegativeInfinity }
            / n:number() _ "*" _ "-" _ "INFINITY" { Expression::NegativeInfinity }

        rule expression() -> Expression = precedence! {
            x:(@) _ "<" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::LessThan, Box::new(y)) }
            --
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            x:(@) _ "-" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Subtract, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            x:(@) _ "/" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Divide, Box::new(y)) }
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
            = n:$(['0'..='9']+ "." ['0'..='9']* "f"?) {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
            / n:$(['0'..='9']+ "f") {
                let n = n.trim_end_matches('f');
                Expression::FloatLiteral(n.parse::<f32>().unwrap())
            }
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
    }
}
