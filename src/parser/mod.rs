pub mod unified_ast;
pub use unified_ast::*;
pub(crate) mod cuda_grammar;
pub(crate) mod host_grammar;

#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

use cuda_grammar::cuda_parser;
use host_grammar::host_parser;

/// Detects and separates host and device code in CUDA source
fn split_cuda_source(source: &str) -> (Vec<String>, Vec<String>) {
    let mut host_parts = Vec::new();
    let mut device_parts = Vec::new();

    // First, remove comments and includes
    let lines: Vec<&str> = source
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("//")
                && !trimmed.starts_with("#include")
                && !trimmed.starts_with("/*")
                && !trimmed.ends_with("*/")
        })
        .collect();

    let cleaned_source = lines.join("\n");

    // Split on __global__ and process each part
    let parts: Vec<String> = cleaned_source
        .split("__global__")
        .map(|s| s.to_string())
        .collect();

    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            if !part.trim().is_empty() {
                host_parts.push(part.trim().to_string());
            }
            continue;
        }

        // Find matching closing brace for the kernel
        let mut brace_count = 0;
        let mut kernel_end = 0;

        for (pos, ch) in part.char_indices() {
            match ch {
                '{' => brace_count += 1,
                '}' => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        kernel_end = pos + 1;
                        break;
                    }
                }
                _ => continue,
            }
        }

        if kernel_end > 0 {
            let kernel = &part[..kernel_end];
            device_parts.push(kernel.to_string());

            // Add remaining code to host parts
            let remaining = &part[kernel_end..];
            if !remaining.trim().is_empty() {
                host_parts.push(remaining.trim().to_string());
            }
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
        let kernel_source = format!("__global__{}", part);
        println!("Attempting to parse kernel:\n{}", kernel_source);

        let kernel = cuda_parser::kernel_function(&kernel_source).map_err(|e| {
            println!("Parser error: {}", e);
            ParserError::DeviceCodeError(e.to_string())
        })?;
        device_code.push(kernel);
    }

    // Parse host code
    let mut host_statements = Vec::new();
    for part in host_parts {
        if !part.trim().is_empty() {
            match host_parser::host_program(&part) {
                Ok(host_code) => host_statements.extend(host_code.statements),
                Err(e) => println!("Warning: Failed to parse host code part: {}", e),
            }
        }
    }

    Ok(CudaProgram {
        device_code,
        host_code: HostCode {
            statements: host_statements,
        },
    })
}
