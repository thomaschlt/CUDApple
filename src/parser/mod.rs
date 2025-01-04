pub mod ast;
mod grammar;
pub mod host_ast;
mod host_grammar;
#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

pub use ast::*;
pub(crate) use grammar::cuda_parser;
pub(crate) use host_grammar::host_parser;

pub fn parse_cuda(input: &str) -> anyhow::Result<CudaProgram> {
    grammar::cuda_parser::program(input)
        .map_err(|e| anyhow::anyhow!("Failed to parse CUDA program: {}", e))
}
