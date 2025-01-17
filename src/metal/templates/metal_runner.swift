import Metal
import Foundation

class MetalKernelRunner {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private var buffers: [String: MTLBuffer] = [:]

    init() throws {
        print("\n=== Starting Metal Device Detection ===")
        
        // Check if Metal framework is available
        print("Checking Metal framework...")
        let metalFrameworkPath = "/System/Library/Frameworks/Metal.framework"
        if FileManager.default.fileExists(atPath: metalFrameworkPath) {
            print("✓ Metal framework found at: \(metalFrameworkPath)")
        } else {
            print("⚠️ Metal framework not found!")
            throw MetalError.deviceNotFound
        }
        
        // Try to get devices
        print("\nAttempting to get Metal devices...")
        let devices = MTLCopyAllDevices()
        print("Number of devices found: \(devices.count)")
        
        if devices.isEmpty {
            print("⚠️ No Metal devices found!")
            throw MetalError.deviceNotFound
        }
        
        // Print device information
        for (index, device) in devices.enumerated() {
            print("\nDevice \(index) Details:")
            print("Name: \(device.name)")
            print("Headless: \(device.isHeadless)")
            print("Low Power: \(device.isLowPower)")
        }
        
        // Try different methods to get a working device
        print("\nTrying different methods to get a working device...")
        
        var selectedDevice: MTLDevice?
        
        // Method 1: Try system default device
        print("\nMethod 1: Trying MTLCreateSystemDefaultDevice()...")
        if let defaultDevice = MTLCreateSystemDefaultDevice() {
            print("✓ Successfully created default device: \(defaultDevice.name)")
            selectedDevice = defaultDevice
        } else {
            print("⚠️ MTLCreateSystemDefaultDevice() failed")
        }
        
        // Method 2: Try to find M1/M2 device
        if selectedDevice == nil {
            print("\nMethod 2: Looking for Apple Silicon device...")
            if let appleDevice = devices.first(where: { $0.name.contains("Apple") }) {
                print("✓ Found Apple Silicon device: \(appleDevice.name)")
                selectedDevice = appleDevice
            } else {
                print("⚠️ No Apple Silicon device found")
            }
        }
        
        // Method 3: Use first available device
        if selectedDevice == nil {
            print("\nMethod 3: Using first available device...")
            if let firstDevice = devices.first {
                print("✓ Using first available device: \(firstDevice.name)")
                selectedDevice = firstDevice
            }
        }
        
        guard let device = selectedDevice else {
            print("\n⚠️ All methods failed to create a Metal device!")
            throw MetalError.deviceNotFound
        }
        
        self.device = device
        print("\nSuccessfully initialized Metal device: \(device.name)")
        
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
                    device uint32_t* n [[buffer(3)]],
                    uint32_t index [[thread_position_in_grid]]) {
                    if (index < *n) {
                        c[index] = a[index] + b[index];
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

    func vectorAdd(a: [Float], b: [Float]) throws -> [Float] {
        let n = a.count
        guard n == b.count else { throw MetalError.invalidBufferSize }
        
        // Allocate buffers
        guard let bufferA = self.allocateBuffer(a, index: 0),
            let bufferB = self.allocateBuffer(b, index: 1),
            let bufferC = self.device.makeBuffer(length: n * MemoryLayout<Float>.stride, 
                                            options: .storageModeShared),
            let bufferN = self.allocateBuffer([UInt32(n)], index: 3) else {
            throw MetalError.bufferAllocationFailed
        }
        
        // Run kernel
        try self.run(inputs: [bufferA, bufferB, bufferC, bufferN], problemSize: n)
        
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