pub mod cuda_grammar;
pub mod unified_ast;

#[cfg(test)]
mod tests;

use crate::parser::cuda_grammar::cuda_parser;
use crate::parser::unified_ast::CudaProgram;
use anyhow::Result;

pub fn parse_cuda(source: &str) -> Result<CudaProgram> {
    let kernel = cuda_parser::kernel_function(source)
        .map_err(|e| anyhow::anyhow!("Failed to parse kernel: {}", e.to_string()))?;

    Ok(CudaProgram {
        device_code: vec![kernel],
    })
}
