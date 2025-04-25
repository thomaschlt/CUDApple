use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use parser::unified_ast::{Expression, KernelFunction, Operator, Statement, Type};

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
    #[arg(short = 'd', long)]
    output: PathBuf,

    // optimization level (0-3)
    #[arg(short, long, default_value_t = 1)]
    opt_level: u8,

    // verbose mode (-v, -vv, -vvv)
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    // run the kernel after generation
    #[arg(short, long)]
    run: bool,
}

fn validate_input(path: &PathBuf) -> Result<()> {
    // Check if file exists
    if !path.exists() {
        anyhow::bail!("Input file does not exist: {:?}", path);
    }

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

fn determine_kernel_dimensions(kernel: &KernelFunction) -> u32 {
    let has_mn_params = kernel
        .parameters
        .iter()
        .any(|p| matches!(p.param_type, Type::Int) && (p.name == "M" || p.name == "N"));

    fn has_matrix_indexing(expr: &Expression) -> bool {
        match expr {
            Expression::BinaryOp(lhs, op, rhs) => match op {
                Operator::Add => {
                    if let Expression::BinaryOp(_, Operator::Multiply, _) = **lhs {
                        true
                    } else {
                        has_matrix_indexing(lhs) || has_matrix_indexing(rhs)
                    }
                }
                _ => has_matrix_indexing(lhs) || has_matrix_indexing(rhs),
            },
            _ => false,
        }
    }

    for stmt in &kernel.body.statements {
        match stmt {
            Statement::Assign(assign) => {
                if has_matrix_indexing(&assign.value) {
                    return 2;
                }
            }
            Statement::ForLoop { .. } => {
                return 2;
            }
            _ => continue,
        }
    }

    if has_mn_params {
        2
    } else {
        1
    }
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
        "Successfully parsed CUDA program with {} kernels",
        cuda_program.device_code.len(),
    );

    // Log kernels
    for kernel in &cuda_program.device_code {
        log::info!("Found kernel: {}", kernel.name);
    }

    // Generate Metal shader
    let mut metal_shader = metal::MetalShader::new();

    // Create config once
    let dimensions = determine_kernel_dimensions(&cuda_program.device_code[0]);
    let config = metal::host::MetalKernelConfig {
        dimensions,
        grid_size: match dimensions {
            1 => (4096, 1, 1),
            2 => {
                let thread_group_size = 16;
                let m = 1000;
                let n = 1000;
                (
                    ((n + thread_group_size - 1) / thread_group_size),
                    ((m + thread_group_size - 1) / thread_group_size),
                    1,
                )
            }
            _ => panic!("Unsupported dimensions"),
        },
        threadgroup_size: match dimensions {
            1 => (256, 1, 1),
            2 => (16, 16, 1),
            _ => panic!("Unsupported dimensions"),
        },
    };

    // Use config for Metal shader
    metal_shader.set_config(config.clone());
    metal_shader
        .generate(&cuda_program)
        .map_err(|e| anyhow::anyhow!("Failed to generate Metal shader: {}", e))?;

    // Write Metal shader
    let shader_path = args.output.join("kernel.metal");
    fs::write(&shader_path, metal_shader.source()).context("Failed to write Metal shader")?;
    log::info!("Generated Metal shader: {:?}", shader_path);

    // Generate Swift host code using the same config
    let kernel = &cuda_program.device_code[0];
    let host_generator = metal::host::MetalHostGenerator::new(
        config, // Reuse the same config here
        metal_shader.source().to_string(),
        kernel.clone(),
    );
    let (swift_runner_code, swift_main_code) = host_generator.generate_swift_code();

    // Write Swift host code
    let host_path = args.output.join("MetalKernelRunner.swift");
    fs::write(&host_path, swift_runner_code).context("Failed to write Swift host code")?;

    let main_path = args.output.join("main.swift");
    fs::write(&main_path, swift_main_code).context("Failed to write Swift main code")?;

    log::info!("Generated Swift files: {:?}, {:?}", host_path, main_path);

    // If --run flag is present, compile and execute the kernel
    if args.run {
        // Run the Swift program
        let run_result = std::process::Command::new("xcrun")
            .current_dir(&args.output)
            .args([
                "-sdk",
                "macosx",
                "swiftc",
                "MetalKernelRunner.swift",
                "main.swift",
                "-o",
                "kernel_runner",
            ])
            .output()
            .context("Failed to compile Swift code")?;

        if !run_result.status.success() {
            println!(
                "Swift compilation error: {}",
                String::from_utf8_lossy(&run_result.stderr)
            );
            anyhow::bail!("Swift compilation failed");
        }

        // Execute the compiled program
        let execute_result = std::process::Command::new("./kernel_runner")
            .current_dir(&args.output)
            .output()
            .context("Failed to run Metal kernel")?;

        if !execute_result.status.success() {
            println!(
                "Execution error: {}",
                String::from_utf8_lossy(&execute_result.stderr)
            );
            anyhow::bail!("Kernel execution failed");
        }

        println!(
            "Kernel execution output:\n{}",
            String::from_utf8_lossy(&execute_result.stdout)
        );
    }

    Ok(())
}
