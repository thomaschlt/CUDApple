import Metal
import Foundation

class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    init() throws {
        print("Attempting to create Metal device...")
        
        // Synchronously initialize Metal
        let semaphore = DispatchSemaphore(value: 0)
        var initDevice: MTLDevice?
        var initError: Error?
        
        DispatchQueue.main.async {
            if let device = MTLCreateSystemDefaultDevice() {
                initDevice = device
            }
            semaphore.signal()
        }
        
        _ = semaphore.wait(timeout: .now() + 2.0)
        
        guard let device = initDevice else {
            throw MetalError.deviceNotFound
        }
        
        self.device = device
        print("Successfully created Metal device: \(device.name)")
        
        // Create command queue with debug info
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            throw MetalError.commandQueueCreationFailed
        }
        print("Successfully created command queue")
        self.commandQueue = commandQueue
        
        // Create pipeline state
        let library = try device.makeLibrary(source: """
                #include <metal_stdlib>
                #include <metal_math>
                using namespace metal;

                kernel void vectorAdd(
                    device float* a [[buffer(0)]],
                    device float* b [[buffer(1)]],
                    device float* c [[buffer(2)]],
                    uint n,
                    uint index [[thread_position_in_grid]]) {
                    uint i = index;
                    if (i < n) {
                        c[i] = a[i] + b[i];
                    }
                }
                """, options: nil)
        print("Successfully created Metal library")
            
        guard let function = library.makeFunction(name: "vectorAdd") else {
            print("Failed to create Metal function")
            throw MetalError.functionNotFound
        }
        print("Successfully created Metal function")
        
        self.pipeline = try device.makeComputePipelineState(function: function)
        print("Successfully created compute pipeline")
    }
    
    func allocateBuffer<T>(_ data: [T], index: Int) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(bytes: data,
                                           length: MemoryLayout<T>.stride * data.count,
                                           options: .storageModeShared) else {
            return nil
        }
        return buffer
    }
    
    func run(inputs: [MTLBuffer]) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        // Set buffers
        for (index, buffer) in inputs.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        // Configure grid and threadgroup sizes
        let gridSize = MTLSize(width: {{GRID_SIZE}}.0,
                             height: {{GRID_SIZE}}.1,
                             depth: {{GRID_SIZE}}.2)
        
        let threadgroupSize = MTLSize(width: {{THREADGROUP_SIZE}}.0,
                                    height: {{THREADGROUP_SIZE}}.1,
                                    depth: {{THREADGROUP_SIZE}}.2)
        
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func vectorAdd(a: [Float], b: [Float]) throws -> [Float] {
        let n = a.count
        guard n == b.count else { throw MetalError.invalidBufferSize }
        
        // Allocate buffers
        guard let bufferA = self.allocateBuffer(a, index: 0),
            let bufferB = self.allocateBuffer(b, index: 1),
            let bufferC = self.device.makeBuffer(length: n * MemoryLayout<Float>.stride, 
                                            options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed
        }
        
        // Run kernel
        try self.run(inputs: [bufferA, bufferB, bufferC])
        
        // Read result
        let result = UnsafeBufferPointer<Float>(start: bufferC.contents().assumingMemoryBound(to: Float.self),
                                        count: n)
        return Array(result)
    } 
}

enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case functionNotFound
    case encoderCreationFailed
    case invalidBufferSize
    case bufferAllocationFailed
}

struct KernelConfig {
    let gridSize: MTLSize
    let threadGroupSize: MTLSize
    
    static func calculate(problemSize: Int) -> KernelConfig {
        let threadGroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let gridWidth = (problemSize + 255) / 256
        let gridSize = MTLSize(width: gridWidth, height: 1, depth: 1)
        return KernelConfig(gridSize: gridSize, threadGroupSize: threadGroupSize)
    }
}