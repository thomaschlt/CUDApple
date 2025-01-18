import Metal
import Foundation

class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    init() throws {
        print("\n=== Metal Device Detection ===")
        
        let devices = MTLCopyAllDevices()
        guard !devices.isEmpty else {
            throw MetalError.deviceNotFound
        }
        
        // Try to find Apple Silicon device
        if let selectedDevice = devices.first(where: { $0.name.contains("Apple") }) {
            print("✓ Using device: \(selectedDevice.name)")
            self.device = selectedDevice
        } else {
            throw MetalError.deviceNotFound
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // Create pipeline state from generated shader
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            #include <metal_math>
            using namespace metal;
            
            {{KERNEL_DEFINITIONS}}
            """, options: nil)
            
        guard let function = library.makeFunction(name: "{{KERNEL_NAME}}") else {
            throw MetalError.functionNotFound
        }
        
        self.pipeline = try device.makeComputePipelineState(function: function)
    }
    
    func allocateBuffer<T>(_ data: [T], index: Int) -> MTLBuffer? {
        guard let buffer = device.makeBuffer(bytes: data,
                                           length: MemoryLayout<T>.stride * data.count,
                                           options: .storageModeShared) else {
            return nil
        }
        return buffer
    }
    
    func run(inputs: [MTLBuffer], problemSize: Int) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.encoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        
        // Set buffers
        for (index, buffer) in inputs.enumerated() {
            computeEncoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        
        // Calculate grid and threadgroup sizes dynamically
        let config = KernelConfig.calculate(problemSize: problemSize)
        
        computeEncoder.dispatchThreads(config.gridSize, 
                                     threadsPerThreadgroup: config.threadGroupSize)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    func executeKernel<T>(inputs: [(data: Any, type: Any.Type)], outputType: T.Type) throws -> [T] {
        // Validate inputs
        guard !inputs.isEmpty else { throw MetalError.invalidInput }
        
        // Allocate buffers dynamically
        var buffers: [MTLBuffer] = []
        for input in inputs.enumerated() {
            if let array = input.1.data as? [T],
               let buffer = device.makeBuffer(length: MemoryLayout<T>.stride * array.count,
                                            options: .storageModeShared) {
                let rawPointer = buffer.contents()
                memcpy(rawPointer, array, array.count * MemoryLayout<T>.stride)
                buffers.append(buffer)
            } else {
                throw MetalError.bufferAllocationFailed
            }
        }
        
        // Get problem size from first input array
        let problemSize = (inputs[0].data as? [T])?.count ?? 0
        
        try self.run(inputs: buffers, problemSize: problemSize)
        return readOutput(buffer: buffers.last!, type: T.self)
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