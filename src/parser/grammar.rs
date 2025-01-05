use crate::parser::unified_ast::*;

peg::parser! {
    pub(crate) grammar cuda_parser() for str {
        rule _() = [' ' | '\t' | '\n' | '\r']*

        rule identifier() -> String
            = s:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                s.to_string()
            }

        rule type_name() -> Type
            = base:base_type() _ pointer:pointer_suffix()? {
                match pointer {
                    Some(_) => Type::Pointer(Box::new(base)),
                    None => base
                }
            }

        rule base_type() -> Type
            = "void" { Type::Void }
            / "int" { Type::Int }
            / "float" { Type::Float }

        rule pointer_suffix() -> ()
            = "*" _ { () }

        rule parameter() -> Parameter
            = t:type_name() _ n:identifier() {
                Parameter {
                    name: n,
                    param_type: t,
                }
            }

        rule thread_index() -> Expression
            = "threadIdx" "." "x" { Expression::ThreadIdx(Dimension::X) }
            / "blockIdx" "." "x" { Expression::BlockIdx(Dimension::X) }
            / "blockDim" "." "x" { Expression::BlockDim(Dimension::X) }

        rule primary() -> Expression
            = n:number() { n }
            / t:thread_index() { t }
            / i:identifier() array_suffix:array_access_suffix()* {
                array_suffix.into_iter().fold(
                    Expression::Variable(i),
                    |acc, expr| Expression::ArrayAccess {
                        array: Box::new(acc),
                        index: Box::new(expr)
                    }
                )
            }
            / "(" _ e:expression() _ ")" { e }

        rule array_access_suffix() -> Expression
            = _ "[" _ expr:expression() _ "]" { expr }

        rule multiplicative() -> Expression
            = head:primary() tail:(_ "*" _ expr:primary() { expr })* {
                tail.into_iter().fold(head, |acc, expr| {
                    Expression::BinaryOp(Box::new(acc), Operator::Multiply, Box::new(expr))
                })
            }

        rule additive() -> Expression
            = head:multiplicative() tail:(_ "+" _ expr:multiplicative() { expr })* {
                tail.into_iter().fold(head, |acc, expr| {
                    Expression::BinaryOp(Box::new(acc), Operator::Add, Box::new(expr))
                })
            }

        rule expression() -> Expression
            = l:additive() _ "<" _ r:additive() { Expression::BinaryOp(Box::new(l), Operator::LessThan, Box::new(r)) }
            / additive()

        rule number() -> Expression
            = n:$(['0'..='9']+) { Expression::IntegerLiteral(n.parse().unwrap()) }

        rule array_access() -> Expression
            = array:identifier() _ "[" _ index:expression() _ "]" {
                Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(array)),
                    index: Box::new(index),
                }
            }

        rule kernel_declaration() -> KernelFunction
            = "__global__" _ "void" _ name:identifier()
              "(" _ params:(parameter() ** (_ "," _)) _ ")" _
              body:block() {
                KernelFunction {
                    name,
                    parameters: params,
                    body,
                }
            }

        rule block() -> Block
            = "{" _ stmts:(statement() ** _) _ "}" {
                Block {
                    statements: stmts
                }
            }

        rule statement() -> Statement
            = decl:declaration() _ ";" { Statement::VariableDecl(decl) }
            / assign:assignment() _ ";" { Statement::Assign(assign) }
            / "if" _ "(" _ cond:expression() _ ")" _ body:block() {
                Statement::IfStmt {
                    condition: cond,
                    body
                }
            }

        rule declaration() -> Declaration
            = t:type_name() _ n:identifier() _ "=" _ e:expression() {
                Declaration {
                    var_type: t,
                    name: n,
                    initializer: Some(e)
                }
            }
            / t:type_name() _ n:identifier() {
                Declaration {
                    var_type: t,
                    name: n,
                    initializer: None
                }
            }

        rule assignment() -> Assignment
            = target:expression() _ "=" _ value:expression() {
                Assignment {
                    target,
                    value
                }
            }

        pub(crate) rule program() -> CudaProgram
            = _ kernels:kernel_declaration()* _ {
                CudaProgram {
                    device_code: kernels,
                    host_code: HostCode { statements: Vec::new() },
                }
            }
    }
}
