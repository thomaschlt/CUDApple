#[cfg(test)]
mod tests {
    use crate::parser::host_grammar::host_parser;
    use crate::parser::unified_ast::{Expression, HostStatement, Type};

    #[test]
    fn test_basic_host_program() {
        let input = r#"
        int n = 1024;
        size_t size = n * sizeof(float);
        float *d_a;
        cudaMalloc(&d_a, size);
        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    "#;
        let result = host_parser::host_program(input);
        if let Err(e) = &result {
            println!("Parse error: {}", e);
        }
        assert!(result.is_ok());
        let host_code = result.unwrap();
        assert!(!host_code.statements.is_empty());
    }

    #[test]
    fn test_simple_declaration() {
        let input = "int n = 1024;";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_cuda_malloc() {
        let input = "cudaMalloc(d_a, size);"; // Missing &
        let result = host_parser::host_program(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_kernel_launch() {
        let input = "vectorAdd<<gridSize, blockSize>>(d_a, d_b, d_c, n);"; // Missing <
        let result = host_parser::host_program(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_assignment() {
        let input = "x = 42;";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
        let host_code = result.unwrap();
        match &host_code.statements[0] {
            HostStatement::Assignment(assignment) => {
                assert!(matches!(assignment.target, Expression::Variable(ref name) if name == "x"));
                assert!(matches!(assignment.value, Expression::IntegerLiteral(42)));
            }
            _ => panic!("Expected Assignment statement"),
        }
    }

    #[test]
    fn test_device_synchronize() {
        let input = "cudaDeviceSynchronize();";
        let result = host_parser::host_program(input);
        assert!(result.is_ok());
        let host_code = result.unwrap();
        assert!(matches!(
            host_code.statements[0],
            HostStatement::DeviceSynchronize
        ));
    }
}
