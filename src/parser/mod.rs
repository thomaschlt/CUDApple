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

fn identify_code_blocks(source: &str) -> Result<Vec<CodeBlock>, ParserError> {
    let (host_parts, device_parts) = split_cuda_source(source);
    let mut blocks = Vec::new();

    // Add host blocks
    for (i, content) in host_parts.iter().enumerate() {
        blocks.push(CodeBlock {
            kind: BlockKind::Host,
            content: content.clone(),
            location: Location {
                start_line: i, // Simplified line tracking
                end_line: i,
            },
        });
    }

    // Add device blocks
    for (i, content) in device_parts.iter().enumerate() {
        blocks.push(CodeBlock {
            kind: BlockKind::Device,
            content: format!("__global__{}", content),
            location: Location {
                start_line: i, // Simplified line tracking
                end_line: i,
            },
        });
    }

    Ok(blocks)
}

pub fn parse_cuda(source: &str) -> Result<CudaProgram, ParserError> {
    // Phase 1: Initial parse to identify all code blocks
    let code_blocks = identify_code_blocks(source)?;

    // Phase 2: Parse each block with appropriate parser
    let mut program = CudaProgram {
        host_code: HostCode {
            statements: Vec::new(),
        },
        device_code: Vec::new(),
    };

    for block in code_blocks {
        match block.kind {
            BlockKind::Host => {
                let host_ast = host_parser::host_program(&block.content)?;
                program.host_code.statements.extend(host_ast.statements);
            }
            BlockKind::Device => {
                let kernel = cuda_parser::kernel_function(&block.content)?;
                program.device_code.push(kernel);
            }
        }
    }

    Ok(program)
}

#[derive(Debug)]
pub struct CodeBlock {
    pub kind: BlockKind,
    pub content: String,
    pub location: Location,
}

#[derive(Debug)]
pub enum BlockKind {
    Host,
    Device,
}

#[derive(Debug)]
pub struct Location {
    pub start_line: usize,
    pub end_line: usize,
}
