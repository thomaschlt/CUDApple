pub mod unified_ast;
pub use unified_ast::*;
mod grammar;
pub(crate) mod host_grammar;

#[cfg(test)]
mod host_tests;
#[cfg(test)]
mod tests;

pub fn parse_cuda(source: &str) -> anyhow::Result<CudaProgram> {
    grammar::cuda_parser::program(source)
        .map_err(|e| anyhow::anyhow!("Failed to parse CUDA program: {}", e))
}
