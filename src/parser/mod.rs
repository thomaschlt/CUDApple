pub mod unified_ast;
pub use unified_ast::*;
pub(crate) mod cuda_grammar;
pub(crate) mod host_grammar;

#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

use cuda_grammar::cuda_parser;

/// Detects and separates host and device code in CUDA source
fn split_cuda_source(source: &str) -> (Vec<&str>, Vec<&str>) {
    let mut host_parts = Vec::new();
    let mut device_parts = Vec::new();

    // Device code is identified by __global__ keyword
    for part in source.split("__global__") {
        if part.trim().is_empty() {
            continue;
        }
        if part.contains("void") {
            device_parts.push(part);
        } else {
            host_parts.push(part);
        }
    }

    (host_parts, device_parts)
}

use crate::parser::unified_ast::ParserError;
use anyhow::Context;

pub fn parse_cuda(source: &str) -> Result<CudaProgram, ParserError> {
    println!("Input source:\n{}", source);
    let (host_parts, device_parts) = split_cuda_source(source);

    println!("Device parts after split:");
    for (i, part) in device_parts.iter().enumerate() {
        println!("Part {}: {}", i, part);
    }

    // Parse device code
    let mut device_code = Vec::new();
    for part in device_parts {
        // Add back the __global__ keyword that was removed during split
        let kernel_source = format!("__global__{}", part);
        println!("Attempting to parse kernel:\n{}", kernel_source);

        let kernel = cuda_parser::kernel_function(&kernel_source).map_err(|e| {
            println!("Parser error: {}", e);
            ParserError::DeviceCodeError(e.to_string())
        })?;
        device_code.push(kernel);
    }

    Ok(CudaProgram {
        device_code,
        host_code: HostCode {
            statements: Vec::new(),
        },
    })
}
