use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub mod metal;
pub mod parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // input: CUDA file
    #[arg(short, long)]
    input: PathBuf,

    // output directory for generated Metal code
    #[arg(short, long)]
    output: PathBuf,

    // optimization level (0-3)
    #[arg(short, long, default_value_t = 1)]
    opt_level: u8,

    // verbose mode (-v, -vv, -vvv)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

fn validate_input(path: &PathBuf) -> Result<()> {
    // Check if file exists
    if !path.exists() {
        anyhow::bail!("Input file does not exist: {:?}", path);
    }

    // Check if it's a file
    if !path.is_file() {
        anyhow::bail!("Input path is not a file: {:?}", path);
    }

    // Check extension
    match path.extension() {
        Some(ext) if ext == "cu" => Ok(()),
        _ => anyhow::bail!("Input file must have .cu extension: {:?}", path),
    }
}

fn ensure_output_dir(path: &PathBuf) -> Result<()> {
    if path.exists() {
        if !path.is_dir() {
            anyhow::bail!("Output path exists but is not a directory: {:?}", path);
        }
    } else {
        fs::create_dir_all(path).context("Failed to create output directory")?;
        log::debug!("Created output directory: {:?}", path);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logger
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            2 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        })
        .init();

    // Validate input file
    validate_input(&args.input).context("Input validation failed")?;

    // Ensure output directory exists
    ensure_output_dir(&args.output).context("Output directory setup failed")?;

    // Log arguments
    log::info!("Processing CUDA file: {:?}", args.input);
    log::info!("Output directory: {:?}", args.output);
    log::info!("Optimization level: {}", args.opt_level);

    // Read and parse the CUDA file
    let cuda_source = fs::read_to_string(&args.input).context("Failed to read CUDA source file")?;

    let cuda_program = parser::parse_cuda(&cuda_source).context("Failed to parse CUDA program")?;

    log::debug!(
        "Successfully parsed CUDA program with {} kernels and {} host statements",
        cuda_program.kernels.len(),
        cuda_program.host_statements.len()
    );

    // Log kernels
    for kernel in &cuda_program.kernels {
        log::info!("Found kernel: {}", kernel.name);
    }

    // Log host statements
    for stmt in &cuda_program.host_statements {
        log::debug!("Found host statement: {:?}", stmt);
    }

    // Generate Metal shader
    let mut metal_shader = metal::MetalShader::new();
    metal_shader.generate(&cuda_program)?;

    // Write Metal shader to output file
    let shader_path = args.output.join("kernel.metal");
    fs::write(&shader_path, metal_shader.source()).context("Failed to write Metal shader")?;

    log::info!("Generated Metal shader: {:?}", shader_path);

    // Generate and write host code
    let config = metal::host::MetalKernelConfig {
        grid_size: (1, 1, 1),          // Default values for now
        threadgroup_size: (256, 1, 1), // Common threadgroup size for 1D
        buffer_sizes: std::collections::HashMap::new(),
    };

    let host_code = metal_shader.generate_host_code(config);
    let host_path = args.output.join("KernelRunner.swift");
    fs::write(&host_path, host_code).context("Failed to write Metal host code")?;

    log::info!("Generated Metal host code: {:?}", host_path);

    Ok(())
}
