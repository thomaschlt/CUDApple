use crate::parser::ast::{Expression, HostStatement, MemcpyKind, Operator, Type};

peg::parser! {
    pub grammar host_parser() for str {
        // Whitespace and comments
        rule _() = [' ' | '\t' | '\n' | '\r']* comment()*
        rule comment() = "//" (!"\n" [_])* "\n" / "/*" (!"*/" [_])* "*/"

        // Basic building blocks
        rule identifier() -> String
            = id:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                id.to_string()
            }

        rule number() -> i64
            = n:$("-"? ['0'..='9']+) {? n.parse().or(Err("invalid number")) }

        // Types
        rule type_name() -> Type
            = "int" { Type::Int }
            / "float" { Type::Float }
            / "void" { Type::Void }

        // Variable declarations
        rule variable_declaration() -> HostStatement
            = t:type_name() _ "*" _ name:identifier() {
                HostStatement::VariableDeclaration {
                    var_type: Type::Pointer(Box::new(t)),
                    name,
                }
            }
            / t:type_name() _ name:identifier() _ "=" _ value:expression() {
                HostStatement::Assignment {
                    variable: name.clone(),
                    value,
                }
            }
            / t:type_name() _ name:identifier() {
                HostStatement::VariableDeclaration {
                    var_type: t,
                    name,
                }
            }

        // Memory operations
        rule cuda_malloc() -> HostStatement
            = "cudaMalloc" _ "(" _ "&" _ ptr:identifier() _ "," _ size:expression() _ ")" {
                HostStatement::MemoryAllocation {
                    variable: ptr,
                    size: size
                }
            }

        rule cuda_memcpy() -> HostStatement
            = "cudaMemcpy" _ "("
                _ dst:identifier() _ ","
                _ src:identifier() _ ","
                _ size:expression() _ ","
                _ dir:memcpy_direction() _
              ")" {
                HostStatement::MemoryCopy {
                    dst,
                    src,
                    size,
                    direction: dir
                }
            }

        rule memcpy_direction() -> MemcpyKind
            = "cudaMemcpyHostToDevice" { MemcpyKind::HostToDevice }
            / "cudaMemcpyDeviceToHost" { MemcpyKind::DeviceToHost }
            / "cudaMemcpyDeviceToDevice" { MemcpyKind::DeviceToDevice }

        // Kernel launch
        rule kernel_launch() -> HostStatement
            = kernel:identifier() _ "<<<" _
              grid:expression() _ "," _
              block:expression() _ ">>>" _
              "(" _ args:kernel_args() _ ")" {
                HostStatement::KernelLaunch {
                    kernel,
                    grid_dim: (grid, Expression::IntegerLiteral(1), Expression::IntegerLiteral(1)),
                    block_dim: (block, Expression::IntegerLiteral(1), Expression::IntegerLiteral(1)),
                    arguments: args
                }
            }

        rule kernel_args() -> Vec<Expression>
            = args:expression() ** (_ "," _) { args }

        // Expressions
        rule expression() -> Expression = precedence! {
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            --
            sizeof:sizeof_expr() { sizeof }
            n:number() { Expression::IntegerLiteral(n) }
            i:identifier() { Expression::Variable(i) }
            "(" _ e:expression() _ ")" { e }
        }

        // Program structure
        pub rule host_program() -> Vec<HostStatement>
            = _
              statements:(host_statement() ** _)
              _ {
                statements
            }

        rule host_statement() -> HostStatement
            = s:(cuda_malloc() /
                cuda_memcpy() /
                kernel_launch() /
                variable_declaration()) _ ";" {
                s
            }

        // Add sizeof operator
        rule sizeof_expr() -> Expression
            = "sizeof" _ "(" _ t:identifier() _ ")" {
                Expression::SizeOf(t)
            }

        // Update initializer rule
        rule initializer() -> Expression
            = expression()
    }
}
