// Unit tests for parser internals

#[cfg(test)]
mod tests {
    use crate::parser::{parse_cuda, Type};

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
}
