use crate::parser::unified_ast::{
    Assignment, Declaration, Expression, HostStatement, MemcpyKind, Operator, Type,
};
use crate::parser::HostCode;

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
            / "cudaEvent_t" { Type::CudaEventT }
            / "dim3" { Type::Dim3 }

        rule pointer_type() -> Type
            = t:base_type() _ "*" { Type::Pointer(Box::new(t)) }

        // Variable declarations
        rule variable_declaration() -> HostStatement
            = type_name:type_specifier() _ first:identifier() rest:(_ "," _ i:identifier() { i })* _ ";" {
                if rest.is_empty() {
                    HostStatement::Declaration(Declaration {
                        var_type: type_name.clone(),
                        name: first.to_string(),
                        initializer: None
                    })
                } else {
                    HostStatement::MultiDeclaration {
                        var_type: type_name,
                        names: {
                            let mut names = vec![first.to_string()];
                            names.extend(rest.into_iter().map(|s| s.to_string()));
                            names
                        }
                    }
                }
            }
            / type_name:type_specifier() _ name:identifier() _ "=" _ value:expression() _ ";" {
                HostStatement::Declaration(Declaration {
                    var_type: type_name,
                    name: name.to_string(),
                    initializer: Some(value)
                })
            }

        // Memory operations
        rule cuda_malloc() -> HostStatement
            = "cudaMalloc" _ "(" _ ptr:malloc_pointer() _ "," _ size:expression() _ ")" _ ";" {
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
            = name:identifier() _ "<<<" _
              grid:dim3_expr() _ "," _
              block:dim3_expr() _ ">>>" _
              "(" _ args:comma_list() _ ")" {
                HostStatement::KernelLaunch {
                    kernel: name,
                    grid_dim: grid,
                    block_dim: block,
                    arguments: args
                }
            }

        // Expressions
        rule expression() -> Expression = precedence! {
            x:(@) _ "+" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Add, Box::new(y)) }
            --
            x:(@) _ "*" _ y:@ { Expression::BinaryOp(Box::new(x), Operator::Multiply, Box::new(y)) }
            --
            x:(@) _ "." _ y:identifier() { Expression::MemberAccess(Box::new(x), y) }
            --
            f:function_call() { f }
            sizeof:sizeof_expr() { sizeof }
            n:number() { Expression::IntegerLiteral(n) }
            i:identifier() { Expression::Variable(i) }
            "(" _ e:expression() _ ")" { e }
        }

        // Helper rule for comma-separated expressions
        rule comma_list() -> Vec<Expression>
            = e:expression() ** (_ "," _) { e }

        // Program structure with better whitespace handling
        pub rule host_program() -> HostCode
            = _ s:host_statement()* _ {
                HostCode { statements: s }
            }

        // Assignment rule
        rule assignment() -> HostStatement
            = var:identifier() _ "=" _ val:expression() _ ";" {
                HostStatement::Assignment(Assignment {
                    target: Expression::Variable(var.to_string()),
                    value: val
                })
            }

        // Device synchronize rule
        rule device_synchronize() -> HostStatement
            = "cudaDeviceSynchronize" _ "(" _ ")" {
                HostStatement::DeviceSynchronize
            }

        // Add CUDA_CHECK macro support
        rule cuda_check_macro() -> HostStatement =
            "CUDA_CHECK" _ "(" _ expr:cuda_check_expr() _ ")" _ ";" {
                HostStatement::MacroCall {
                    name: "CUDA_CHECK".to_string(),
                    arguments: vec![expr]
                }
            }

        // Add CUDA Event operations
        rule cuda_event_operation() -> HostStatement
            = event_create() / event_record() / event_synchronize() / event_elapsed_time() / event_destroy()

        rule event_create() -> HostStatement
            = "cudaEventCreate" _ "(" _ "&" _ name:identifier() _ ")" {
                HostStatement::EventCreate { event: name }
            }

        rule event_record() -> HostStatement
            = "cudaEventRecord" _ "(" _ event:identifier() _ ")" {
                HostStatement::EventRecord { event }
            }

        rule event_synchronize() -> HostStatement
            = "cudaEventSynchronize" _ "(" _ event:identifier() _ ")" {
                HostStatement::EventSynchronize { event }
            }

        rule event_elapsed_time() -> HostStatement
            = "cudaEventElapsedTime" _ "(" _ "&" _ var:identifier() _ "," _
              start:identifier() _ "," _ stop:identifier() _ ")" {
                HostStatement::EventElapsedTime {
                    milliseconds: var,
                    start: start,
                    end: stop
                }
            }

        rule event_destroy() -> HostStatement
            = "cudaEventDestroy" _ "(" _ event:identifier() _ ")" {
                HostStatement::EventDestroy { event }
            }

        // Update the host_statement rule to include cuda_event_operation
        pub rule host_statement() -> HostStatement
            = _ s:(
                cuda_check_macro() /
                variable_declaration() /
                assignment() /
                dim3_declaration() /
                cuda_malloc() /
                cuda_memcpy() /
                cuda_free() /
                device_synchronize() /
                cuda_event_operation()
            ) _ { s }

        // Add sizeof operator
        rule sizeof_expr() -> Expression
            = "sizeof" _ "(" _ t:type_name() _ ")" {
                Expression::SizeOf(t)
            }

        // Update initializer rule
        rule initializer() -> Expression
            = expression()

        rule dim3_expr() -> (Expression, Expression, Expression) =
            "dim3" _ "(" _
            x:expression() _
            y:("," _ e:expression() { e })? _
            z:("," _ e:expression() { e })? _
            ")" {
                (
                    x,
                    y.unwrap_or(Expression::IntegerLiteral(1)),
                    z.unwrap_or(Expression::IntegerLiteral(1))
                )
            }
            / x:expression() {
                (
                    x,
                    Expression::IntegerLiteral(1),
                    Expression::IntegerLiteral(1)
                )
            }

        rule cuda_free() -> HostStatement
            = "cudaFree" _ "(" _ var:identifier() _ ")" {
                HostStatement::MemoryFree { variable: var }
            }

        rule string_literal() -> String
            = "\"" s:$([^'"']*) "\"" { s.to_string() }

        rule parameter_list() -> Vec<(Type, String)>
            = first:(t:type_name() _ name:identifier() { (t, name) })
              rest:(_ "," _ t:type_name() _ name:identifier() { (t, name) })* {
                let mut params = vec![first];
                params.extend(rest);
                params
            }

        rule block() -> Vec<HostStatement>
            = "{" _ s:host_statement()* "}" {
                s
            }

        rule argument_list() -> Vec<Expression>
            = first:expression() rest:(_ "," _ e:expression() { e })* {
                let mut args = vec![first];
                args.extend(rest);
                args
            }

        rule dim3_declaration() -> HostStatement =
            "dim3" _ name:identifier() _ "(" _
            x:expression() _
            y:("," _ e:expression() { e })? _
            z:("," _ e:expression() { e })? _
            ")" _ ";" {
                HostStatement::Dim3Declaration {
                    name: name.to_string(),
                    x,
                    y: Some(y.unwrap_or(Expression::IntegerLiteral(1))),
                    z: Some(z.unwrap_or(Expression::IntegerLiteral(1)))
                }
            }

        rule type_specifier() -> Type =
            t:base_type() _ "*" { Type::Pointer(Box::new(t)) }
            / base_type()

        rule function_call() -> Expression
            = name:identifier() _ "(" _ args:argument_list() _ ")" {
                Expression::FunctionCall(name, args)
            }

        // Add cuda_api_call rule
        rule cuda_api_call() -> Expression =
            name:identifier() _ "(" _ args:argument_list() _ ")" {
                Expression::FunctionCall(name, args)
            }

        rule cuda_check_expr() -> Expression =
            event_create_expr() /
            event_destroy_expr() /
            event_record_expr() /
            event_synchronize_expr()

        rule event_create_expr() -> Expression =
            "cudaEventCreate" _ "(" _ "&" _ name:identifier() _ ")" {
                Expression::FunctionCall(
                    "cudaEventCreate".to_string(),
                    vec![Expression::AddressOf(Box::new(Expression::Variable(name)))]
                )
            }

        rule event_destroy_expr() -> Expression =
            "cudaEventDestroy" _ "(" _ event:identifier() _ ")" {
                Expression::FunctionCall(
                    "cudaEventDestroy".to_string(),
                    vec![Expression::Variable(event)]
                )
            }

        rule event_record_expr() -> Expression =
            "cudaEventRecord" _ "(" _ event:identifier() _ ")" {
                Expression::FunctionCall(
                    "cudaEventRecord".to_string(),
                    vec![Expression::Variable(event)]
                )
            }

        rule event_synchronize_expr() -> Expression =
            "cudaEventSynchronize" _ "(" _ event:identifier() _ ")" {
                Expression::FunctionCall(
                    "cudaEventSynchronize".to_string(),
                    vec![Expression::Variable(event)]
                )
            }
    }
}
