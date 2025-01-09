use crate::metal::MetalShader;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_cuda;

    fn print_metal_translation(cuda_source: &str) {
        let cuda_program = parse_cuda(cuda_source).unwrap();
        let mut metal_shader = MetalShader::new();
        metal_shader.generate(&cuda_program).unwrap();

        println!("Generated Metal:\n{}\n", metal_shader.source());
    }

    #[test]
    fn test_vector_add_translation() {
        let cuda_source = r#"__global__ void vectorAdd(float *a, float *b, float *c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }"#;
        print_metal_translation(cuda_source);
    }

    #[test]
    fn test_simple_array_access() {
        let cuda_source = r#"__global__ void arrays(float *arr) {
            arr[0] = 1;
            arr[threadIdx.x] = 2;
        }"#;
        print_metal_translation(cuda_source);
    }

    #[test]
    fn test_thread_indices() {
        let cuda_source = r#"__global__ void indices() {
            int tx = threadIdx.x;
            int bx = blockIdx.x;
            int dx = blockDim.x;
        }"#;
        print_metal_translation(cuda_source);
    }

    #[test]
    fn test_arithmetic() {
        let cuda_source = r#"__global__ void math() {
            int a = 1 + 2;
            int b = 3 * 4;
            int c = a + b;
        }"#;
        print_metal_translation(cuda_source);
    }

    #[test]
    fn test_math_functions() {
        let cuda_source = r#"__global__ void math_test(float *out) {
            int i = threadIdx.x;
            float x = -1.0f * INFINITY;
            float y = max(x, 0.0f);
            out[i] = expf(y);
        }"#;
        print_metal_translation(cuda_source);
    }

    #[test]
    fn test_kernel_timing() {
        let cuda_source = r#"
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            kernel<<<grid, block>>>(args);
            cudaEventRecord(stop);
            
            cudaEventSynchronize(stop);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        "#;
        print_metal_translation(cuda_source);
    }
}
