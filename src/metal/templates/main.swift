import Foundation

{{PARAMETER_INIT}}

// After parameter initialization
print("\n=== Parameter Initialization ===")
print("Problem Size: \(problemSize)")

do {
    // Initialize our Metal kernel runner
    let runner = try MetalKernelRunner()
    
    print("\n=== Running {{KERNEL_NAME}} ({{DIMENSION_TYPE}} kernel) ===")
    {{KERNEL_CALL}}
    
    print("\n=== Kernel Results: {{KERNEL_NAME}} ===")
    print("Input values shown above")
    print("\nOutput values:")
    for i in 0..<5 {
        print("result[\(i)] = \(result[i])")
    }
} catch {
    print("Error: \(error)")
}
