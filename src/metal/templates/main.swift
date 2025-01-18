import Foundation

{{PARAMETER_INIT}}

do {
    // Initialize our Metal kernel runner
    let runner = try MetalKernelRunner()
    
    {{KERNEL_CALL}}
    
    print("\n=== Kernel Results: {{KERNEL_NAME}} ===")
    for i in 0..<5 {
        print("result[\(i)] = \(result[i])")
    }
} catch {
    print("Error: \(error)")
}