// Unit tests for parser internals

#[cfg(test)]
mod tests {
    use crate::parser::{
        parse_cuda,
        unified_ast::{ParserError, Qualifier, Type},
        Expression, Operator, Statement,
    };

    #[test]
    fn test_basic_kernel_declaration() {
        let input = r#"__global__ void simple() {}"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        assert_eq!(program.device_code[0].name, "simple");
        assert_eq!(program.device_code[0].parameters.len(), 0);
        println!("\nAST Structure:\n{:?}", program);
    }

    #[test]
    fn test_parameter_types() {
        let input = r#"__global__ void types(int a, float b, int *c, float *d) {}"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        let params = &program.device_code[0].parameters;
        assert_eq!(params[0].param_type, Type::Int);
        assert_eq!(params[1].param_type, Type::Float);
        assert_eq!(params[2].param_type, Type::Pointer(Box::new(Type::Int)));
        assert_eq!(params[3].param_type, Type::Pointer(Box::new(Type::Float)));
        println!("\nAST Structure:\n{:?}", program);
    }

    #[test]
    fn test_variable_declaration() {
        let input = r#"__global__ void vars() {
            int a;
            float b;
            int *c;
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        let statements = &program.device_code[0].body.statements;
        assert_eq!(statements.len(), 3);
        println!("\nAST Structure:\n{:?}", program);
    }

    #[test]
    fn test_variable_initialization() {
        let input = r#"__global__ void init() {
            int a = 42;
            int b = a;
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_arithmetic_operations() {
        let input = r#"__global__ void math() {
            int a = 1 + 2;
            int b = 3 * 4;
            int c = a + b;
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_array_access() {
        let input = r#"__global__ void arrays(float *arr) {
            arr[0] = 1;
            arr[threadIdx.x] = 2;
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_thread_indices() {
        let input = r#"__global__ void indices() {
            int tx = threadIdx.x;
            int bx = blockIdx.x;
            int dx = blockDim.x;
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        let statements = &program.device_code[0].body.statements;
        assert_eq!(statements.len(), 3);
        println!("\nAST Structure:\n{:?}", program);
    }

    #[test]
    fn test_if_statement() {
        let input = r#"__global__ void conditional(int n) {
            if (threadIdx.x < n) {
                int a = 1;
            }
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_complex_expression() {
        let input = r#"__global__ void complex(float *a, float *b) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            a[i] = a[i] + b[i];
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_assignment() {
        let input = r#"__global__ void assign() {
            int a;
            a = 1;
            a = a + 1;      
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_vector_add() {
        let input = r#"__global__ void vectorAdd(float *a, float *b, float *c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        println!("\nAST Structure:\n{:?}", result.unwrap());
    }

    #[test]
    fn test_error_handling() {
        // Test syntax error in host code
        let input = r#"
            int n = 1024
            cudaMalloc(&d_a, size);  // Missing semicolon above
        "#;
        let result = parse_cuda(input);
        assert!(matches!(result, Err(ParserError::HostCodeError(_))));

        // Test syntax error in device code
        let input = r#"
            __global__ void kernel( {  // Missing closing parenthesis
                int i = 0;
            }
        "#;
        let result = parse_cuda(input);
        assert!(matches!(result, Err(ParserError::DeviceCodeError(_))));
    }

    #[test]
    fn test_basic_restrict() {
        let input = r#"__global__ void simple(float* __restrict__ a) {
            // Empty kernel body
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        assert_eq!(
            program.device_code[0].parameters[0].qualifier,
            Qualifier::Restrict
        );
        println!("\nParsed AST:\n{:?}", program);
    }

    #[test]
    fn test_restrict_qualifier() {
        let input = r#"__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
            int row = blockDim.x * blockIdx.x + threadIdx.x;

            if (row < M) {
                // max
                float m = -1 * INFINITY;
                // norm factor
                float L = 0.0f;

                // 3 passes (not optimal)
                for (int col = 0; col < N; col++) {
                    int i = row * N + col;
                    m = max(m, matd[i]);
                }
                for (int col = 0; col < N; col++) {
                    int i = row * N + col;
                    L += expf(matd[i] - m);
                }
                for (int col = 0; col < N; col++) {
                    int i = row * N + col;
                    resd[i] = expf(matd[i] - m) / L;
                }
            }
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        assert_eq!(
            program.device_code[0].parameters[0].qualifier,
            Qualifier::Restrict
        );
    }

    #[test]
    fn test_parser_debug() {
        let input = r#"__global__ void simple(float* __restrict__ a) {}"#;
        println!("\n=== Testing Parser ===");
        println!("Input: {}", input);
        let result = parse_cuda(input);
        match result {
            Ok(program) => {
                println!("Success! Parsed AST:");
                println!("{:#?}", program);
                assert_eq!(
                    program.device_code[0].parameters[0].qualifier,
                    Qualifier::Restrict
                );
            }
            Err(e) => {
                println!("Parser failed: {:?}", e);
                panic!("Parser should not fail");
            }
        }
    }

    #[test]
    fn test_for_loop() {
        let input = r#"__global__ void loop_test() {
            for (int i = 0; i < 10; i = i + 1) {
                int x = i * 2;
            }
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();

        // Get the first statement from the kernel body
        let statements = &program.device_code[0].body.statements;
        assert_eq!(statements.len(), 1, "Expected one for-loop statement");

        // Verify it's a ForLoop
        if let Statement::ForLoop {
            init,
            condition,
            increment: _,
            body,
        } = &statements[0]
        {
            // Check initialization
            if let Statement::VariableDecl(decl) = &**init {
                assert_eq!(decl.name, "i");
                assert_eq!(decl.var_type, Type::Int);
            } else {
                panic!("Expected variable declaration as loop initializer");
            }

            // Check condition
            if let Expression::BinaryOp(_, op, _) = condition {
                assert!(matches!(op, Operator::LessThan));
            } else {
                panic!("Expected binary operation as condition");
            }

            // Check body contains one statement
            assert_eq!(
                body.statements.len(),
                1,
                "Expected one statement in loop body"
            );
        } else {
            panic!("Expected ForLoop statement");
        }

        println!("\nParsed AST:\n{:#?}", program);
    }

    #[test]
    fn test_for_loop_with_assignment() {
        let input = r#"__global__ void loop_test() {
            for (int i = 0; i < 10; i = i + 1) {
                int x = i * 2;
            }
        }"#;
        let result = parse_cuda(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let program = result.unwrap();
        println!("\nParsed AST:\n{:#?}", program);
    }

}
