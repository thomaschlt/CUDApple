#[cfg(test)]
mod tests {
    use crate::parser::host_grammar::host_parser::{self, host_program};
    use crate::parser::unified_ast::{Expression, HostStatement, Type};

    // !fail
    #[test]
    fn test_basic_host_program() {
        let input = r#"
        int n = 1024;
        size_t size = n * sizeof(float);
        float *d_a;
        dim3 block_size(1024);
        dim3 grid_size(CEIL_DIV(M, block_size.x));
        cudaEvent_t start, stop;
        CUDA_CHECK();
    "#;
        let result = host_parser::host_program(input);
        if let Err(e) = &result {
            println!("Parse error: {}", e);
        }
        assert!(result.is_ok());
        let host_code = result.unwrap();
        println!("\nAST Structure:\n{:#?}", host_code);
        assert!(!host_code.statements.is_empty());
    }

    // !work
    #[test]
    fn test_simple_declaration() {
        let input = "int n = 1024;";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
        println!("\nAST Structure:\n{:#?}", result.unwrap());
    }

    #[test]
    fn test_block_size() {
        let input = "dim3 block_size(1024);";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
        println!("\nAST Structure:\n{:#?}", result.unwrap());
    }

    // !strange
    #[test]
    fn test_invalid_kernel_launch() {
        let input = "vectorAdd<<gridSize, blockSize>>(d_a, d_b, d_c, n);";
        let result = host_parser::host_program(input);
        println!("\nParse Result:\n{:#?}", result);
        assert!(result.is_err());
    }

    // !work
    #[test]
    fn test_assignment() {
        let input = "x = 42;";
        let result = host_parser::host_program(input);
        assert!(result.is_ok(), "Parsing failed: {:?}", result);
        let host_code = result.unwrap();
        println!("\nAST Structure:\n{:#?}", host_code);
        match &host_code.statements[0] {
            HostStatement::Assignment(assignment) => {
                assert!(matches!(assignment.target, Expression::Variable(ref name) if name == "x"));
                assert!(matches!(assignment.value, Expression::IntegerLiteral(42)));
            }
            _ => panic!("Expected Assignment statement"),
        }
    }

    // !fail
    #[test]
    fn test_device_synchronize() {
        let input = "cudaDeviceSynchronize();";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
        let host_code = result.unwrap();
        println!("\nAST Structure:\n{:#?}", host_code);
        assert!(matches!(
            host_code.statements[0],
            HostStatement::DeviceSynchronize
        ));
    }

    // !fail
    #[test]
    fn test_dim3_declaration() {
        let inputs = [
            "dim3 block_size(1024);",
            "dim3 grid_size(CEIL_DIV(M, block_size.x));",
            "dim3 threads(256, 1, 1);",
        ];

        for input in inputs {
            let result = host_parser::host_program(input);
            assert!(result.is_ok(), "Failed to parse: {}", input);
            println!("\nInput: {}", input);
            println!("AST Structure:\n{:#?}", result.as_ref().unwrap());

            if let Ok(host_code) = result {
                match &host_code.statements[0] {
                    HostStatement::Dim3Declaration { name, x, y, z } => {
                        assert!(!name.is_empty());
                        match input {
                            "dim3 block_size(1024);" => {
                                assert!(matches!(x, Expression::IntegerLiteral(1024)));
                                assert!(y.is_none());
                                assert!(z.is_none());
                            }
                            "dim3 threads(256, 1, 1);" => {
                                assert!(matches!(x, Expression::IntegerLiteral(256)));
                                assert!(matches!(y, Some(Expression::IntegerLiteral(1))));
                                assert!(matches!(z, Some(Expression::IntegerLiteral(1))));
                            }
                            _ => {
                                assert!(matches!(x, Expression::FunctionCall { .. }));
                            }
                        }
                    }
                    _ => panic!("Expected Dim3Declaration"),
                }
            }
        }
    }

    // !work
    #[test]
    fn test_pointer_declaration() {
        let input = "float *d_a;";
        let result = host_parser::host_program(input);
        assert!(result.is_ok(), "Failed to parse: {}", input);
        println!("\nInput: {}", input);
        println!("AST Structure:\n{:#?}", result.as_ref().unwrap());

        if let Ok(host_code) = result {
            match &host_code.statements[0] {
                HostStatement::Declaration(decl) => match &decl.var_type {
                    Type::Pointer(_) => assert!(true),
                    _ => panic!("Expected Declaration"),
                },
                _ => panic!("Expected Declaration"),
            }
        }
    }
}
