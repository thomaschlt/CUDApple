pub mod unified_ast;
pub use unified_ast::*;
pub(crate) mod cuda_grammar;
mod grammar;
pub(crate) mod host_grammar;

#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

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
    let (host_parts, device_parts) = split_cuda_source(source);

    // Parse host code
    let host_code = host_parts
        .iter()
        .map(|part| {
            host_grammar::host_parser::host_program(part)
                .map_err(|e| ParserError::HostCodeError(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(
            HostCode {
                statements: Vec::new(),
            },
            |mut acc, code| {
                acc.statements.extend(code.statements);
                acc
            },
        );

    // Parse device code
    let device_code = device_parts
        .iter()
        .map(|part| {
            cuda_grammar::cuda_parser::kernel_function(part)
                .map_err(|e| ParserError::DeviceCodeError(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CudaProgram {
        host_code,
        device_code,
    })
}
