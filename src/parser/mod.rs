mod ast;
mod grammar;
#[cfg(test)]
mod tests;

pub use ast::*;
pub(crate) use grammar::cuda_parser;

pub fn parse_cuda(input: &str) -> anyhow::Result<CudaProgram> {
    cuda_parser::program(input).map_err(|e| anyhow::anyhow!("Parse error: {}", e))
}
