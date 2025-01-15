use crate::metal::MetalShader;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_cuda;

    #[test]
    fn test_vector_add_translation() {
        let cuda_source = r#"
        __global__ void vectorAdd(float *a, float *b, float *c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }"#;

        let program = parse_cuda(cuda_source).unwrap();
        let mut shader = MetalShader::new();
        shader.generate(&program).unwrap();


        let metal_source = shader.source();
        // Verify basic Metal translation
        assert!(metal_source.contains("kernel void vectorAdd"));
        assert!(metal_source.contains("device float* a [[buffer(0)]]"));
        assert!(metal_source.contains("uint index [[thread_position_in_grid]]"));
    }
}
