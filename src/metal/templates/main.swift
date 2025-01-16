import Foundation

// Test data
let n = 1000000
var a = Array(repeating: Float(0), count: n)
var b = Array(repeating: Float(0), count: n)

// Initialize test data
for i in 0..<n {
    a[i] = Float(i)
    b[i] = Float(i * 2)
}

do {
    // Initialize our Metal kernel runner
    let runner = try MetalKernelRunner()
    
    // Run vector addition
    let result = try runner.vectorAdd(a: a, b: b)
    
    // Verify first few results
    for i in 0..<5 {
        print("result[\(i)] = \(result[i])")
    }
} catch {
    print("Error: \(error)")
}
