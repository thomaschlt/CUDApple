use crate::parser::unified_ast::{
    Declaration, Expression, HostStatement, MemcpyKind, Operator, Type,
};

peg::parser! {
    pub grammar host_parser() for str {
        // Whitespace handling
        rule _() = quiet!{([' ' | '\t' | '\n' | '\r'] / comment())*}

        rule comment() = block_comment() / line_comment()
        rule block_comment() = "/*" (!"*/" [_])* "*/"
        rule line_comment() = "//" (!"\n" [_])* ("\n" / ![_])

        // Basic building blocks
        rule identifier() -> String
            = id:$(['a'..='z' | 'A'..='Z' | '_']['a'..='z' | 'A'..='Z' | '0'..='9' | '_']*) {
                id.to_string()
            }

        rule number() -> i64
            = n:$("-"? ['0'..='9']+) {? n.parse().or(Err("invalid number")) }

        // Types
        rule type_name() -> Type
            = pointer_type() / base_type()

        rule base_type() -> Type
            = "int" { Type::Int }
            / "float" { Type::Float }
            / "void" { Type::Void }
            / "size_t" { Type::SizeT }

        rule pointer_type() -> Type
            = t:base_type() _ "*" { Type::Pointer(Box::new(t)) }

        // Variable declarations
        rule variable_declaration() -> HostStatement
            = t:type_name() _ name:identifier() _ "=" _ value:expression() {
                HostStatement::Declaration(Declaration {
                    var_type: t,
                    name: name.clone(),
                    initializer: Some(value)
                })
            }
            / t:type_name() _ name:identifier() {
                HostStatement::Declaration(Declaration {
                    var_type: t,
                    name,
                    initializer: None
                })
            }

        // Memory operations
        rule cuda_malloc() -> HostStatement
            = "cudaMalloc" _ "(" _ ptr:malloc_pointer() _ "," _ size:expression() _ ")" {
                HostStatement::MemoryAllocation {
                    variable: ptr,
                    size
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

        // Helper rule to enforce & operator
        rule malloc_pointer() -> String
            = "&" _ name:identifier() { name }

        // Kernel launch with strict <<< >>> syntax
        rule kernel_launch() -> HostStatement
            = name:identifier() _ "<<<" _ grid:expression() _ "," _ block:expression() _ ">>>"
              _ "(" _ args:comma_list() _ ")" {
                HostStatement::KernelLaunch {
                    kernel: name,
                    grid_dim: (grid, Expression::IntegerLiteral(1), Expression::IntegerLiteral(1)),
                    block_dim: (block, Expression::IntegerLiteral(1), Expression::IntegerLiteral(1)),
                    arguments: args
                }
            }

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

        // Helper rule for comma-separated expressions
        rule comma_list() -> Vec<Expression>
            = e:expression() ** (_ "," _) { e }

        // Program structure with better whitespace handling
        pub rule host_program() -> Vec<HostStatement>
            = _ statements:host_statement() ** (_ ";" _) _ ";"? _ {
                statements
            }

        rule host_statement() -> HostStatement
            = _ s:(
                cuda_malloc() /
                cuda_memcpy() /
                kernel_launch() /
                variable_declaration()
            ) _ {
                s
            }

        // Add sizeof operator
        rule sizeof_expr() -> Expression
            = "sizeof" _ "(" _ t:type_name() _ ")" {
                Expression::SizeOf(t)
            }

        // Update initializer rule
        rule initializer() -> Expression
            = expression()
    }
}
