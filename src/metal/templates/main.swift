import Foundation

print("\n=== CUDApple Kernel Execution ===")
print("• Emulating CUDA kernel: {{KERNEL_NAME}}")

{{PARAMETER_INIT}}

do {
    // Initialize our Metal kernel runner
    let runner = try MetalKernelRunner()
    
    let startTime = CFAbsoluteTimeGetCurrent()
    {{KERNEL_CALL}}
    let endTime = CFAbsoluteTimeGetCurrent()
    
    print("• Kernel execution completed in \(String(format: "%.3f", (endTime - startTime) * 1000))ms")
    
    print("\n=== Results ===")
    print("• First 5 output values:")
    for i in 0..<5 {
        print("  [\(i)]: \(result[i])")
    }
} catch {
    print("\n[ERROR] \(error)")
}
