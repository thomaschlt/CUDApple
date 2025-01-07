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
            / "__restrict" { Qualifier::Restrict }
            / { Qualifier::None }

        rule parameter() -> Parameter
            = param_type:type_specifier() _ "*"? _ qual:qualifier()? _ name:identifier() {
                Parameter {
                    param_type,
                    name: name.to_string(),
                    qualifier: qual.unwrap_or(Qualifier::None)
                }
            }

        rule parameter_list() -> Vec<Parameter>
            = first:parameter() rest:(_ "," _ p:parameter() { p })* {
                let mut params = vec![first];
                params.extend(rest);
                params
            }

        rule block() -> Block
            = "{" _ stmts:statement()* _ "}" {
                Block { statements: stmts }
            }

        rule variable_declaration() -> Statement
            = var_type:type_specifier() _ name:identifier() _ init:("=" _ e:expression() { e })? _ ";" {
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
            = target:identifier() _ "=" _ value:expression() {
                Statement::Assign(Assignment {
                    target: Expression::Variable(target),
                    value
                })
            }

        rule if_statement() -> Statement
            = "if" _ "(" _ condition:expression() _ ")" _ body:block() {
                Statement::IfStmt { condition, body }
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
                if_statement() /
                for_loop() /
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
            n:number() { Expression::Number(n) }
            i:identifier() { Expression::Variable(i) }
            array_access()
            "(" _ e:expression() _ ")" { e }
        }
        rule number() -> i64 = n:$(['0'..='9']+) {
            n.parse().unwrap()
        }

        rule for_loop() -> Statement
            = "for" _ "(" _
              init:(variable_declaration() / assignment()) _ ";" _
              condition:expression() _ ";" _
              increment:assignment() _ ")" _
              body:block() {
                Statement::ForLoop {
                    init: Box::new(init),
                    condition,
                    increment: Box::new(increment),
                    body
                }
            }
    }
}
