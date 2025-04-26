import Metal
import Foundation

class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    init() throws {
        print("\n=== Metal Device Detection ===")
        print("• Scanning for compatible Metal devices...")
        
        let devices = MTLCopyAllDevices()
        guard !devices.isEmpty else {
            throw MetalError.deviceNotFound
        }
        
        // Try to find Apple Silicon device
        if let selectedDevice = devices.first(where: { $0.name.contains("Apple") }) {
            print("• Using device: \(selectedDevice.name)")
            print("  ├─ Recommended max threads per threadgroup: \(selectedDevice.maxThreadsPerThreadgroup)")
            print("  └─ Supports unified memory: \(selectedDevice.hasUnifiedMemory ? "Yes" : "No")")
            self.device = selectedDevice
        } else {
            throw MetalError.deviceNotFound
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // Create pipeline state from generated shader
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = false
        compileOptions.languageVersion = .version2_4

        let library = try device.makeLibrary(
            source: """
            #include <metal_stdlib>
            #include <metal_math>
            using namespace metal;
            
            {{KERNEL_DEFINITIONS}}
            """, 
            options: compileOptions)

        if let error = library.makeFunction(name: "{{KERNEL_NAME}}")?.label {
            print("Function creation error: \(error)")
        }
        
        self.pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "{{KERNEL_NAME}}")!)
    }
    
    func allocateBuffer<T>(_ data: [T], index: Int) -> MTLBuffer? {
        print("\n• Allocating buffer \(index)")
        print("  ├─ Elements: \(data.count)")
        print("  └─ Size: \(MemoryLayout<T>.stride * data.count) bytes")
        
        guard let buffer = device.makeBuffer(bytes: data,
                                           length: MemoryLayout<T>.stride * data.count,
                                           options: .storageModeShared) else {
            print("[ERROR] Failed to allocate buffer \(index)")
            return nil
        }
        print("• Successfully allocated buffer \(index)")
        return buffer
    }
    
    func run(inputs: [MTLBuffer], problemSize: Int) throws {
        // Add debug prints for input buffers
        print("\n=== Buffer Contents Before Kernel Execution ===")
        for (index, buffer) in inputs.enumerated() {
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            print("Buffer \(index) first 5 elements:")
            for i in 0..<5 {
                print("  [\(i)]: \(ptr[i])")
            }
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        // Set buffers
        for (index, buffer) in inputs.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        let config = KernelConfig.calculate(
            problemSize: problemSize,
            dimensions: {{DIMENSIONS}},
            width: {{WIDTH}},
            height: {{HEIGHT}}
        )
        
        print("\n=== Kernel Configuration ===")
        print("Grid Size: \(config.gridSize)")
        print("Thread Group Size: \(config.threadGroupSize)")
        print("Problem Size: \(problemSize)")
        print("Total Threads: \(config.gridSize.width * config.gridSize.height * config.threadGroupSize.width * config.threadGroupSize.height)")
        
        // Verify configuration
        if (config.gridSize.width * config.gridSize.height * 
            config.threadGroupSize.width * config.threadGroupSize.height) < problemSize {
            print("Warning: Grid size might be insufficient for problem size")
        }
        
        computeEncoder.dispatchThreadgroups(config.gridSize, 
                                     threadsPerThreadgroup: config.threadGroupSize)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        if let error = commandBuffer.error {
            print("Command Buffer Error: \(error)")
            throw MetalError.executionFailed
        }
        
        // Add verification after kernel execution
        print("\n=== Buffer Contents After Kernel Execution ===")
        for (index, buffer) in inputs.enumerated() {
            let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
            print("Buffer \(index) first 5 elements:")
            for i in 0..<5 {
                print("  [\(i)]: \(ptr[i])")
            }
        }
    }

    func executeKernel<T>(inputs: [(data: Any, type: Any.Type)], outputType: T.Type) throws -> [T] {
        // Validate inputs
        guard !inputs.isEmpty else { throw MetalError.invalidInput }
        
        var buffers: [MTLBuffer] = []
        
        let problemSize = if let firstArray = inputs[0].data as? [T] {
            firstArray.count
        } else {
            throw MetalError.invalidInput
        }
        
        // Allocate and copy input data
        for (index, input) in inputs.enumerated() {
            if let array = input.data as? [T] {
                guard let buffer = allocateBuffer(array, index: index) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            } else if let scalar = input.data as? UInt32 {
                guard let buffer = device.makeBuffer(bytes: [scalar],
                                                   length: MemoryLayout<UInt32>.size,
                                                   options: .storageModeShared) else {
                    throw MetalError.bufferAllocationFailed
                }
                buffers.append(buffer)
            }
        }
        
        try self.run(inputs: buffers, problemSize: problemSize)
        
        // Read back the result from the output buffer (res)
        guard buffers.count > 2 else {
            throw MetalError.invalidInput
        }
        
        let outputBuffer = buffers[2] // res is the third buffer
        let outputPtr = outputBuffer.contents().assumingMemoryBound(to: T.self)
        return Array(UnsafeBufferPointer(start: outputPtr, count: problemSize))
    }

    func readOutput<T>(buffer: MTLBuffer, type: T.Type) -> [T] {
        let count = buffer.length / MemoryLayout<T>.stride
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case functionNotFound
    case encoderCreationFailed
    case invalidBufferSize
    case bufferAllocationFailed
    case invalidInput
    case executionFailed
}

struct KernelConfig {
    let gridSize: MTLSize
    let threadGroupSize: MTLSize
    
    static func calculate(problemSize: Int, dimensions: Int, width: Int?, height: Int?) -> KernelConfig {
        if dimensions == 1 {
            let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let gridWidth = (problemSize + 255) / 256
            let gridSize = MTLSize(width: gridWidth, height: 1, depth: 1)
            return KernelConfig(gridSize: gridSize, threadGroupSize: threadGroupSize)
        } else {
            // For 2D matrices, use M and N directly
            guard let w = width, let h = height else {
                fatalError("2D kernels require explicit width (M) and height (N)")
            }
            
            let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let gridSize = MTLSize(
                // width: (w + 15) / 16,  // Ceiling division for M dimension
                // height: (h + 15) / 16, // Ceiling division for N dimension
                // width: (w + threadGroupSize.width - 1) / threadGroupSize.width,
                // height: (h + threadGroupSize.height - 1) / threadGroupSize.height,
                width: w,
                height: h,
                depth: 1
            )
            return KernelConfig(gridSize: gridSize, threadGroupSize: threadGroupSize)
        }
    }
}