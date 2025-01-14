use crate::metal::MetalShader;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_cuda;

    #[test]
    fn test_basic_kernel_translation() {
        let cuda_source = r#"
        __global__ void simple_kernel(float *data) {
            int idx = threadIdx.x;
            data[idx] = data[idx] * 2.0f;
        }"#;

        let program = parse_cuda(cuda_source).unwrap();
        let mut shader = MetalShader::new();
        shader.generate(&program).unwrap();

        println!("Generated Metal:\n{}", shader.source());
    }
}
