use std::collections::HashMap;

#[derive(Debug)]
pub struct MetalKernelConfig {
    pub grid_size: (u32, u32, u32),
    pub threadgroup_size: (u32, u32, u32),
    pub buffer_sizes: HashMap<String, usize>,
}

pub struct MetalHostGenerator {
    config: MetalKernelConfig,
    shader: String,
}

impl MetalHostGenerator {
    pub fn new(config: MetalKernelConfig, shader: String) -> Self {
        Self { config, shader }
    }

    pub fn generate_swift_code(&self) -> String {
        let mut code = String::new();

        // 1. Import statements
        code.push_str("import Metal\n\n");

        // 2. Class definition
        code.push_str("class MetalKernelRunner {\n");

        // 3. Properties
        code.push_str("    private let device: MTLDevice\n");
        code.push_str("    private let commandQueue: MTLCommandQueue\n");
        code.push_str("    private let pipeline: MTLComputePipelineState\n\n");

        // 4. Implementation
        self.generate_init_method(&mut code);
        self.generate_run_method(&mut code);

        code.push_str("}\n");
        code
    }

    fn generate_init_method(&self, code: &mut String) {
        code.push_str(
            r#"    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create command queue")
        }
        self.commandQueue = queue
        
        // Load and compile the kernel
        let library = try! device.makeLibrary(source: """
"#,
        );
        code.push_str(&self.shader);
        code.push_str(
            r#"""
            , options: nil)
            
        guard let function = library.makeFunction(name: "kernel_main") else {
            fatalError("Failed to create kernel function")
        }
        
        self.pipeline = try! device.makeComputePipelineState(function: function)
    }
"#,
        );
    }

    fn generate_run_method(&self, code: &mut String) {
        code.push_str(
            r#"    func run(
        inputs: [MTLBuffer],
        gridSize: MTLSize,
        threadGroupSize: MTLSize
    ) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create compute command encoder")
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        // Bind buffers
        for (index, buffer) in inputs.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        // Dispatch threadgroups
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
"#,
        );
    }
}
