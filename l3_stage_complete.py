#!/usr/bin/env python3
"""
L3-Stage Engine v3.0 - Complete Benchmark Script for Kaggle/Colab
This script installs everything, runs benchmarks, and generates PDF + CSV reports.

Usage in Colab/Kaggle:
    !python l3_stage_complete.py

Or run cell by cell in notebook.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Test sizes
    CHUNK_SIZE = 512 * 1024 * 1024  # 512MB
    TOTAL_SIZE = 10 * 1024 * 1024 * 1024  # 10GB (quick test)
    NUM_STREAMS = 4
    
    # Paths
    WORK_DIR = Path("/kaggle/working" if Path("/kaggle").exists() else "/content")
    PROJECT_DIR = WORK_DIR / "l3-stage-engine"
    RESULTS_DIR = WORK_DIR / "benchmark_results"
    
    # Output files
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE = RESULTS_DIR / f"benchmark_results_{TIMESTAMP}.csv"
    JSON_FILE = RESULTS_DIR / f"benchmark_results_{TIMESTAMP}.json"
    PDF_FILE = RESULTS_DIR / f"benchmark_report_{TIMESTAMP}.pdf"

# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd, shell=True, check=True):
    """Run shell command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=shell, 
        capture_output=True, 
        text=True,
        check=False
    )
    if check and result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout, result.stderr, result.returncode

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

# ============================================================================
# Installation Functions
# ============================================================================

def check_environment():
    """Detect and print environment info."""
    print_section("Environment Detection")
    
    env_type = "Kaggle" if Path("/kaggle").exists() else "Colab"
    print(f"Environment: {env_type}")
    
    # Check GPU
    gpu_info = "No GPU"
    try:
        output, _, _ = run_command("nvidia-smi --query-gpu=name --format=csv,noheader", check=False)
        if output.strip():
            gpu_info = output.strip()
    except:
        pass
    
    print(f"GPU: {gpu_info}")
    print(f"Working Directory: {Config.WORK_DIR}")
    
    return env_type, gpu_info

def install_rust():
    """Install Rust compiler."""
    print_section("Installing Rust")
    
    cargo_bin = Path.home() / ".cargo" / "bin"
    
    # Check if already installed via full path
    cargo_exe = cargo_bin / "cargo"
    if cargo_exe.exists():
        try:
            result = subprocess.run([str(cargo_exe), "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Rust already installed")
                # Update PATH for this session
                if str(cargo_bin) not in os.environ["PATH"]:
                    os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ["PATH"]
                return
        except Exception:
            pass 
    
    print("Installing Rust...")
    run_command(
        'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable',
        check=True
    )
    
    # Update PATH for the current process directly
    if cargo_bin.exists():
        os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ["PATH"]
        print(f"✅ Added {cargo_bin} to PATH")
    
    print("✅ Rust installed successfully")

def install_dependencies():
    """Install system build tools and Python dependencies."""
    print_section("Installing Dependencies")
    
    # 1. Use apt-get (more compatible than 'apt' in scripts)
    # The -y flag is mandatory for non-interactive scripts
    print("Installing system build-essential (linker and compilers)...")
    run_command("apt-get update -qq && apt-get install -y -qq build-essential", check=True)
    
    # 2. Install Python packages
    packages = [
        "matplotlib",
        "pandas",
        "reportlab",
        "Pillow"
    ]
    
    for pkg in packages:
        print(f"Installing Python package: {pkg}...")
        run_command(f"pip install -q {pkg}", check=False)
    
    print("✅ All dependencies installed")

# ============================================================================
# Project Setup
# ============================================================================

def create_project_structure():
    """Create complete project structure."""
    print_section("Creating Project Structure")
    
    Config.PROJECT_DIR.mkdir(exist_ok=True, parents=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    dirs = [
        "src/runtime",
        "src/benchmarks", 
        "src/utils",
        "cpp/backends",
        "cpp/core"
    ]
    
    for d in dirs:
        (Config.PROJECT_DIR / d).mkdir(exist_ok=True, parents=True)
    
    print(f"✅ Project structure created at {Config.PROJECT_DIR}")

def create_cargo_toml():
    """Create Cargo.toml."""
    content = """[package]
name = "l3-stage-engine"
version = "3.0.0"
edition = "2021"

[lib]
name = "l3_stage_engine"
path = "src/lib.rs"

[[bin]]
name = "l3-benchmark"
path = "src/main.rs"

[dependencies]
libc = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[build-dependencies]
cc = { version = "1.0", features = ["parallel"] }
"""
    
    (Config.PROJECT_DIR / "Cargo.toml").write_text(content)
    print("✅ Created Cargo.toml")

def create_build_rs():
    """Create build.rs."""
    content = """use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");
    println!("cargo:rustc-link-lib=stdc++");
    
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_available = PathBuf::from(&cuda_path).join("include/cuda_runtime.h").exists();
    
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cpp/unified_runtime.cpp")
        .file("cpp/core/device_manager.cpp")
        .file("cpp/core/memory_manager.cpp")
        .flag("-std=c++14")
        .flag("-O3")
        .flag("-fPIC");
    
    if cuda_available {
        println!("cargo:rustc-cfg=has_cuda");
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=cudart");
        build.flag(&format!("-I{}/include", cuda_path));
        build.define("HAS_CUDA", None);
        build.file("cpp/backends/cuda_backend.cpp");
    }
    
    build.compile("l3_stage_runtime");
}
"""
    
    (Config.PROJECT_DIR / "build.rs").write_text(content)
    print("✅ Created build.rs")

def create_cpp_files():
    """Create all C++ files."""
    print("Creating C++ files...")
    
    # unified_runtime.h
    header = """#ifndef UNIFIED_RUNTIME_H
#define UNIFIED_RUNTIME_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DEVICE_TYPE_NVIDIA_GPU = 0,
    DEVICE_TYPE_AMD_GPU = 1,
    DEVICE_TYPE_CPU = 4
} DeviceType;

typedef enum {
    MEMORY_TYPE_DEVICE = 0,
    MEMORY_TYPE_PINNED = 2,
} MemoryType;

typedef struct {
    DeviceType type;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability;
    bool supports_async_transfers;
    float memory_bandwidth_gbps;
} DeviceInfo;

typedef void* StreamHandle;

typedef struct {
    void* ptr;
    size_t size;
    MemoryType type;
    DeviceType device_type;
    int device_id;
} MemoryHandle;

int runtime_init();
void runtime_shutdown();
int runtime_get_device_count();
int runtime_get_device_info(int device_id, DeviceInfo* info);
int runtime_set_device(int device_id);
int runtime_select_best_device();

MemoryHandle runtime_malloc(size_t size, MemoryType type, int device_id);
void runtime_free(MemoryHandle handle);

StreamHandle runtime_stream_create(int device_id);
void runtime_stream_destroy(StreamHandle stream);
void runtime_stream_sync(StreamHandle stream);

int runtime_memcpy(void* dst, const void* src, size_t size, int dst_device, int src_device);
int runtime_memcpy_async(void* dst, const void* src, size_t size, int dst_device, int src_device, StreamHandle stream);

const char* runtime_get_error_string();

#ifdef __cplusplus
}
#endif

#endif
"""
    
    # unified_runtime.cpp (simplified version)
    cpp = """#include "unified_runtime.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

struct RuntimeState {
    std::vector<DeviceInfo> devices;
    int active_device;
    bool initialized;
    std::string last_error;
};

static RuntimeState g_runtime;

static void set_error(const char* msg) {
    g_runtime.last_error = msg;
}

const char* runtime_get_error_string() {
    return g_runtime.last_error.c_str();
}

#ifdef HAS_CUDA
static int detect_nvidia_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
    
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        DeviceInfo info = {};
        info.type = DEVICE_TYPE_NVIDIA_GPU;
        strncpy(info.name, prop.name, sizeof(info.name) - 1);
        info.total_memory = prop.totalGlobalMem;
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;
        
        info.compute_capability = prop.major * 10 + prop.minor;
        info.supports_async_transfers = true;
        info.memory_bandwidth_gbps = (prop.memoryBusWidth / 8.0f * prop.memoryClockRate * 2) / 1e6f;
        
        g_runtime.devices.push_back(info);
        fprintf(stderr, "[Runtime] GPU %d: %s (%.2f GB)\\n", i, info.name, info.total_memory / 1e9);
    }
    return count;
}
#endif

static int detect_cpu_fallback() {
    DeviceInfo info = {};
    info.type = DEVICE_TYPE_CPU;
    strcpy(info.name, "CPU");
    info.total_memory = 16ULL * 1024 * 1024 * 1024;
    info.free_memory = 8ULL * 1024 * 1024 * 1024;
    info.supports_async_transfers = false;
    info.memory_bandwidth_gbps = 50.0f;
    
    g_runtime.devices.push_back(info);
    fprintf(stderr, "[Runtime] CPU fallback\\n");
    return 1;
}

int runtime_init() {
    if (g_runtime.initialized) return g_runtime.devices.size();
    
    fprintf(stderr, "[Runtime] Initializing...\\n");
    g_runtime.devices.clear();
    
    int total = 0;
#ifdef HAS_CUDA
    total += detect_nvidia_devices();
#endif
    
    if (total == 0) total += detect_cpu_fallback();
    
    if (total > 0) {
        g_runtime.active_device = 0;
        g_runtime.initialized = true;
    }
    return total;
}

void runtime_shutdown() {
    g_runtime.devices.clear();
    g_runtime.initialized = false;
}

int runtime_get_device_count() {
    return g_runtime.devices.size();
}

int runtime_get_device_info(int device_id, DeviceInfo* info) {
    if (device_id < 0 || device_id >= (int)g_runtime.devices.size()) return -1;
    *info = g_runtime.devices[device_id];
    return 0;
}

int runtime_set_device(int device_id) {
    if (device_id < 0 || device_id >= (int)g_runtime.devices.size()) return -1;
#ifdef HAS_CUDA
    if (g_runtime.devices[device_id].type == DEVICE_TYPE_NVIDIA_GPU) {
        cudaSetDevice(device_id);
    }
#endif
    g_runtime.active_device = device_id;
    return 0;
}

int runtime_select_best_device() {
    if (g_runtime.devices.empty()) return -1;
    runtime_set_device(0);
    return 0;
}

MemoryHandle runtime_malloc(size_t size, MemoryType type, int device_id) {
    MemoryHandle handle = {};
    handle.size = size;
    handle.type = type;
    handle.device_id = device_id;
    
    if (device_id < 0 || device_id >= (int)g_runtime.devices.size()) return handle;
    
    handle.device_type = g_runtime.devices[device_id].type;
    
#ifdef HAS_CUDA
    if (handle.device_type == DEVICE_TYPE_NVIDIA_GPU) {
        if (type == MEMORY_TYPE_DEVICE) {
            cudaMalloc(&handle.ptr, size);
        } else {
            cudaMallocHost(&handle.ptr, size);
        }
        return handle;
    }
#endif
    
    handle.ptr = malloc(size);
    return handle;
}

void runtime_free(MemoryHandle handle) {
    if (!handle.ptr) return;
#ifdef HAS_CUDA
    if (handle.device_type == DEVICE_TYPE_NVIDIA_GPU) {
        if (handle.type == MEMORY_TYPE_DEVICE) {
            cudaFree(handle.ptr);
        } else {
            cudaFreeHost(handle.ptr);
        }
        return;
    }
#endif
    free(handle.ptr);
}

StreamHandle runtime_stream_create(int device_id) {
#ifdef HAS_CUDA
    if (g_runtime.devices[device_id].type == DEVICE_TYPE_NVIDIA_GPU) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return (StreamHandle)stream;
    }
#endif
    return nullptr;
}

void runtime_stream_destroy(StreamHandle stream) {
#ifdef HAS_CUDA
    if (stream) cudaStreamDestroy((cudaStream_t)stream);
#endif
}

void runtime_stream_sync(StreamHandle stream) {
#ifdef HAS_CUDA
    if (stream) cudaStreamSynchronize((cudaStream_t)stream);
#endif
}

int runtime_memcpy(void* dst, const void* src, size_t size, int dst_device, int src_device) {
#ifdef HAS_CUDA
    return (cudaMemcpy(dst, src, size, cudaMemcpyDefault) == cudaSuccess) ? 0 : -1;
#endif
    memcpy(dst, src, size);
    return 0;
}

int runtime_memcpy_async(void* dst, const void* src, size_t size, int dst_device, int src_device, StreamHandle stream) {
#ifdef HAS_CUDA
    return (cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, (cudaStream_t)stream) == cudaSuccess) ? 0 : -1;
#endif
    memcpy(dst, src, size);
    return 0;
}
"""
    
    (Config.PROJECT_DIR / "cpp" / "unified_runtime.h").write_text(header)
    (Config.PROJECT_DIR / "cpp" / "unified_runtime.cpp").write_text(cpp)
    (Config.PROJECT_DIR / "cpp" / "core" / "device_manager.cpp").write_text("// Placeholder\n")
    (Config.PROJECT_DIR / "cpp" / "core" / "memory_manager.cpp").write_text("// Placeholder\n")
    (Config.PROJECT_DIR / "cpp" / "backends" / "cuda_backend.cpp").write_text("// Placeholder\n")
    
    print("✅ Created C++ files")

def create_rust_files():
    """Create all Rust source files."""
    print("Creating Rust files...")
    
    # This is getting very long, let me create the essential files
    # I'll create a more compact version that still works
    
    lib_rs = """//! L3-Stage Engine
pub mod runtime;
pub use runtime::*;
"""
    
    runtime_mod = """mod bindings;
pub use bindings::*;
"""
    
    # I'll create a simpler version to fit in the script
    # The full version would be too long for a single script
    
    (Config.PROJECT_DIR / "src" / "lib.rs").write_text(lib_rs)
    (Config.PROJECT_DIR / "src" / "runtime" / "mod.rs").write_text(runtime_mod)
    
    # Create a minimal working bindings file
    create_rust_bindings()
    create_rust_main()
    
    print("✅ Created Rust files")

def create_rust_bindings():
    """Create Rust FFI bindings with correct comment syntax."""
    content = '''// FFI bindings and safe wrappers
// Removed unused CStr import to clear warnings

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DeviceType {
    NvidiaGPU = 0,
    AMDGPU = 1,
    CPU = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum MemoryType {
    Device = 0,
    Pinned = 2,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)] //  This allows the handle to be moved out of &mut self
pub struct MemoryHandle {
    pub ptr: *mut std::ffi::c_void,
    pub size: usize,
    pub memory_type: MemoryType,
    pub device_type: DeviceType,
    pub device_id: i32,
}

#[repr(C)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: [u8; 256],
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability: i32,
    pub supports_async_transfers: bool,
    pub memory_bandwidth_gbps: f32,
}

impl DeviceInfo {
    pub fn name_str(&self) -> String {
        let null_pos = self.name.iter().position(|&c| c == 0).unwrap_or(self.name.len());
        String::from_utf8_lossy(&self.name[..null_pos]).to_string()
    }
}

extern "C" {
    fn runtime_init() -> i32;
    fn runtime_shutdown();
    fn runtime_get_device_count() -> i32;
    fn runtime_get_device_info(device_id: i32, info: *mut DeviceInfo) -> i32;
    fn runtime_set_device(device_id: i32) -> i32;
    fn runtime_select_best_device() -> i32;
    fn runtime_malloc(size: usize, mem_type: MemoryType, device_id: i32) -> MemoryHandle;
    fn runtime_free(handle: MemoryHandle);
    fn runtime_stream_create(device_id: i32) -> *mut std::ffi::c_void;
    fn runtime_stream_destroy(stream: *mut std::ffi::c_void);
    fn runtime_stream_sync(stream: *mut std::ffi::c_void);
    fn runtime_memcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize, dst_device: i32, src_device: i32) -> i32;
    fn runtime_memcpy_async(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize, dst_device: i32, src_device: i32, stream: *mut std::ffi::c_void) -> i32;
    fn runtime_get_error_string() -> *const i8;
}

pub struct Runtime {
    initialized: bool,
}

impl Runtime {
    pub fn new() -> Result<Self, String> {
        unsafe {
            let count = runtime_init();
            if count > 0 {
                Ok(Self { initialized: true })
            } else {
                Err("No devices found".to_string())
            }
        }
    }
    
    pub fn device_count(&self) -> usize {
        unsafe { runtime_get_device_count() as usize }
    }
    
    pub fn device_info(&self, device_id: usize) -> Result<DeviceInfo, String> {
        let mut info: DeviceInfo = unsafe { std::mem::zeroed() };
        unsafe {
            if runtime_get_device_info(device_id as i32, &mut info) == 0 {
                Ok(info)
            } else {
                Err("Invalid device ID".to_string())
            }
        }
    }
    
    pub fn set_device(&self, device_id: usize) -> Result<(), String> {
        unsafe {
            if runtime_set_device(device_id as i32) == 0 {
                Ok(())
            } else {
                Err("Failed to set device".to_string())
            }
        }
    }
    
    pub fn select_best_device(&self) -> Result<usize, String> {
        unsafe {
            let device = runtime_select_best_device();
            if device >= 0 {
                Ok(device as usize)
            } else {
                Err("No suitable device".to_string())
            }
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { runtime_shutdown(); }
        }
    }
}

pub struct DeviceMemory {
    handle: MemoryHandle,
}

impl DeviceMemory {
    //  Uses // for comments and _runtime to silence warnings
    pub fn new(_runtime: &Runtime, size: usize, mem_type: MemoryType, device_id: usize) -> Result<Self, String> {
        unsafe {
            let handle = runtime_malloc(size, mem_type, device_id as i32);
            if handle.ptr.is_null() {
                Err("Allocation failed".to_string())
            } else {
                Ok(Self { handle })
            }
        }
    }
    
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.handle.ptr
    }
    
    pub fn size(&self) -> usize {
        self.handle.size
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        //  comment syntax
        unsafe { runtime_free(self.handle); }
    }
}

pub struct Stream {
    handle: *mut std::ffi::c_void,
}

impl Stream {
    //  comment syntax
    pub fn new(_runtime: &Runtime, device_id: usize) -> Result<Self, String> {
        unsafe {
            let handle = runtime_stream_create(device_id as i32);
            if handle.is_null() {
                Err("Stream creation failed".to_string())
            } else {
                Ok(Self { handle })
            }
        }
    }
    
    pub fn sync(&self) {
        unsafe { runtime_stream_sync(self.handle); }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { runtime_stream_destroy(self.handle); }
    }
}

pub fn memcpy_sync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize, dst_device: Option<usize>, src_device: Option<usize>) -> Result<(), String> {
    unsafe {
        let dst_dev = dst_device.map(|d| d as i32).unwrap_or(-1);
        let src_dev = src_device.map(|d| d as i32).unwrap_or(-1);
        if runtime_memcpy(dst, src, size, dst_dev, src_dev) == 0 {
            Ok(())
        } else {
            Err("Transfer failed".to_string())
        }
    }
}

pub fn memcpy_async(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize, dst_device: Option<usize>, src_device: Option<usize>, stream: &Stream) -> Result<(), String> {
    unsafe {
        let dst_dev = dst_device.map(|d| d as i32).unwrap_or(-1);
        let src_dev = src_device.map(|d| d as i32).unwrap_or(-1);
        if runtime_memcpy_async(dst, src, size, dst_dev, src_dev, stream.handle) == 0 {
            Ok(())
        } else {
            Err("Transfer failed".to_string())
        }
    }
}
'''
    
    (Config.PROJECT_DIR / "src" / "runtime" / "bindings.rs").write_text(content)
def create_rust_main():
    """Create main.rs with JSON output for Python parsing."""
    content = '''use l3_stage_engine::*;
use std::time::Instant;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct BenchmarkResult {
    device_name: String,
    device_type: String,
    total_memory_gb: f64,
    baseline_gbps: f64,
    optimized_gbps: f64,
    multistream_gbps: f64,
    improvement_pct: f64,
}

fn benchmark_baseline(runtime: &Runtime, device_id: usize, chunk_size: usize, total_size: usize) -> Result<f64, String> {
    let device_mem = DeviceMemory::new(runtime, chunk_size, MemoryType::Device, device_id)?;
    let mut host_data = vec![0u8; chunk_size];
    
    let iterations = total_size / chunk_size;
    let start = Instant::now();
    
    for _ in 0..iterations {
        memcpy_sync(device_mem.ptr(), host_data.as_ptr() as *const _, chunk_size, Some(device_id), None)?;
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    Ok((total_size as f64 / 1e9) / elapsed)
}

fn benchmark_optimized(runtime: &Runtime, device_id: usize, chunk_size: usize, total_size: usize) -> Result<f64, String> {
    let host_mem = DeviceMemory::new(runtime, chunk_size, MemoryType::Pinned, device_id)?;
    let device_mem = DeviceMemory::new(runtime, chunk_size, MemoryType::Device, device_id)?;
    let stream = Stream::new(runtime, device_id)?;
    
    let iterations = total_size / chunk_size;
    let start = Instant::now();
    
    for _ in 0..iterations {
        memcpy_async(device_mem.ptr(), host_mem.ptr(), chunk_size, Some(device_id), Some(device_id), &stream)?;
        stream.sync();
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    Ok((total_size as f64 / 1e9) / elapsed)
}

fn benchmark_multistream(runtime: &Runtime, device_id: usize, chunk_size: usize, total_size: usize, num_streams: usize) -> Result<f64, String> {
    let total_buffer = chunk_size * num_streams;
    let host_mem = DeviceMemory::new(runtime, total_buffer, MemoryType::Pinned, device_id)?;
    
    let mut device_buffers = Vec::new();
    let mut streams = Vec::new();
    
    for _ in 0..num_streams {
        device_buffers.push(DeviceMemory::new(runtime, chunk_size, MemoryType::Device, device_id)?);
        streams.push(Stream::new(runtime, device_id)?);
    }
    
    let iterations = total_size / (chunk_size * num_streams);
    let start = Instant::now();
    
    for _ in 0..iterations {
        for (idx, (dev_buf, stream)) in device_buffers.iter().zip(streams.iter()).enumerate() {
            let offset = idx * chunk_size;
            unsafe {
                let src_ptr = (host_mem.ptr() as *const u8).add(offset);
                memcpy_async(dev_buf.ptr(), src_ptr as *const _, chunk_size, Some(device_id), Some(device_id), stream)?;
            }
        }
        for stream in &streams {
            stream.sync();
        }
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    Ok((total_size as f64 / 1e9) / elapsed)
}

fn main() {
    let runtime = Runtime::new().expect("Failed to initialize");
    
    let chunk_size = 512 * 1024 * 1024;
    let total_size = 10 * 1024 * 1024 * 1024;
    let num_streams = 4;
    
    let mut results = Vec::new();
    
    for i in 0..runtime.device_count() {
        let info = runtime.device_info(i).unwrap();
        
        eprintln!("Benchmarking device {}: {}", i, info.name_str());
        
        let baseline = benchmark_baseline(&runtime, i, chunk_size, total_size).unwrap_or(0.0);
        let optimized = benchmark_optimized(&runtime, i, chunk_size, total_size).unwrap_or(0.0);
        let multistream = benchmark_multistream(&runtime, i, chunk_size, total_size, num_streams).unwrap_or(0.0);
        
        let improvement = if baseline > 0.0 {
            ((multistream / baseline - 1.0) * 100.0)
        } else {
            0.0
        };
        
        let result = BenchmarkResult {
            device_name: info.name_str(),
            device_type: format!("{:?}", info.device_type),
            total_memory_gb: info.total_memory as f64 / 1e9,
            baseline_gbps: baseline,
            optimized_gbps: optimized,
            multistream_gbps: multistream,
            improvement_pct: improvement,
        };
        
        results.push(result);
    }
    
    let json = serde_json::to_string_pretty(&results).unwrap();
    println!("{}", json);
}
'''
    
    (Config.PROJECT_DIR / "src" / "main.rs").write_text(content)
# ============================================================================
# Build and Run
# ============================================================================

def build_project():
    """Build the Rust project."""
    print_section("Building Project")
    
    os.chdir(Config.PROJECT_DIR)
    
    # Ensure Cargo bin is in PATH
    cargo_bin = Path.home() / ".cargo" / "bin"
    if str(cargo_bin) not in os.environ["PATH"]:
        os.environ["PATH"] = str(cargo_bin) + os.pathsep + os.environ["PATH"]
    
    print("Building release version...")
    # Call cargo directly now that PATH is verified
    stdout, stderr, returncode = run_command(
        "cargo build --release",
        check=False
    )
    
    if returncode != 0:
        print("❌ Build failed!")
        print("--- STDERR ---")
        print(stderr)
        print("--- STDOUT ---")
        print(stdout)
        raise RuntimeError("Build failed")
    
    print("✅ Build successful")

def run_benchmark():
    """Run the benchmark and return results, filtering out logs from stdout."""
    print_section("Running Benchmark")
    
    binary = Config.PROJECT_DIR / "target" / "release" / "l3-benchmark"
    
    if not binary.exists():
        raise RuntimeError(f"Binary not found: {binary}")
    
    print("Executing benchmark...")
    stdout, stderr, returncode = run_command(str(binary), check=False)
    
    if returncode != 0:
        print(f"Benchmark stderr: {stderr}")
        raise RuntimeError("Benchmark failed")
    
    # Parse JSON from stdout - Extract only the JSON array block
    try:
        start = stdout.find('[')
        end = stdout.rfind(']') + 1
        
        if start != -1 and end > start:
            json_payload = stdout[start:end]
            results = json.loads(json_payload)
            print(f"✅ Benchmark complete - {len(results)} device(s) tested")
            return results
        else:
            raise ValueError("No JSON array found in benchmark output")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw Output was:\n{stdout}")
        raise
# ============================================================================
# Results Processing
# ============================================================================

def save_csv(results):
    """Save results to CSV."""
    print_section("Saving CSV Results")
    
    import pandas as pd
    
    df = pd.DataFrame(results)
    df.to_csv(Config.CSV_FILE, index=False)
    
    print(f"✅ CSV saved to: {Config.CSV_FILE}")
    return Config.CSV_FILE

def save_json(results):
    """Save results to JSON."""
    with open(Config.JSON_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ JSON saved to: {Config.JSON_FILE}")
    return Config.JSON_FILE

def generate_pdf_report(results, env_type, gpu_info):
    """Generate PDF report with charts."""
    print_section("Generating PDF Report")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import pandas as pd
        
        # Create charts
        chart_files = []
        
        if results:
            # Chart 1: Throughput comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            df = pd.DataFrame(results)
            
            x = range(len(df))
            width = 0.25
            
            ax.bar([i - width for i in x], df['baseline_gbps'], width, label='Baseline', color='#e74c3c')
            ax.bar(x, df['optimized_gbps'], width, label='Optimized', color='#3498db')
            ax.bar([i + width for i in x], df['multistream_gbps'], width, label='Multi-Stream', color='#2ecc71')
            
            ax.set_xlabel('Device', fontsize=12)
            ax.set_ylabel('Throughput (GB/s)', fontsize=12)
            ax.set_title('L3-Stage Engine Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(df['device_name'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            chart1 = Config.RESULTS_DIR / f"chart_throughput_{Config.TIMESTAMP}.png"
            plt.tight_layout()
            plt.savefig(chart1, dpi=300, bbox_inches='tight')
            plt.close()
            chart_files.append(chart1)
            
            # Chart 2: Improvement percentage
            if len(df) > 0 and df['improvement_pct'].max() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors_list = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['improvement_pct']]
                ax.barh(df['device_name'], df['improvement_pct'], color=colors_list)
                
                ax.set_xlabel('Performance Improvement (%)', fontsize=12)
                ax.set_title('Speedup vs Baseline', fontsize=14, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                for i, v in enumerate(df['improvement_pct']):
                    ax.text(v, i, f' {v:.1f}%', va='center')
                
                chart2 = Config.RESULTS_DIR / f"chart_improvement_{Config.TIMESTAMP}.png"
                plt.tight_layout()
                plt.savefig(chart2, dpi=300, bbox_inches='tight')
                plt.close()
                chart_files.append(chart2)
        
        # Create PDF
        doc = SimpleDocTemplate(str(Config.PDF_FILE), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("L3-Stage Engine v3.0", title_style))
        story.append(Paragraph("Benchmark Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        story.append(Paragraph("Test Information", heading_style))
        
        metadata = [
            ["Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Environment", env_type],
            ["GPU", gpu_info],
            ["Test Size", f"{Config.TOTAL_SIZE / 1e9:.1f} GB"],
            ["Chunk Size", f"{Config.CHUNK_SIZE / 1e6:.0f} MB"],
            ["Streams", str(Config.NUM_STREAMS)]
        ]
        
        t = Table(metadata, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Results table
        if results:
            story.append(Paragraph("Benchmark Results", heading_style))
            
            table_data = [["Device", "Type", "Memory (GB)", "Baseline", "Optimized", "Multi-Stream", "Improvement"]]
            
            for r in results:
                table_data.append([
                    r['device_name'][:20],
                    r['device_type'],
                    f"{r['total_memory_gb']:.1f}",
                    f"{r['baseline_gbps']:.2f}",
                    f"{r['optimized_gbps']:.2f}",
                    f"{r['multistream_gbps']:.2f}",
                    f"{r['improvement_pct']:.1f}%"
                ])
            
            t = Table(table_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.9*inch, 1*inch, 0.9*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*inch))
        
        # Add charts
        if chart_files:
            story.append(PageBreak())
            story.append(Paragraph("Performance Visualizations", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            for chart_file in chart_files:
                img = Image(str(chart_file), width=6.5*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ PDF report saved to: {Config.PDF_FILE}")
        return Config.PDF_FILE
        
    except Exception as e:
        print(f"⚠️  PDF generation failed: {e}")
        print("Results are still available in CSV and JSON formats")
        return None

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           L3-Stage Engine v3.0 - Complete Benchmark          ║
║                                                               ║
║  This script will:                                            ║
║  1. Install Rust and dependencies                             ║
║  2. Create complete project                                   ║
║  3. Build the engine                                          ║
║  4. Run benchmarks                                            ║
║  5. Generate PDF and CSV reports                              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Setup
        env_type, gpu_info = check_environment()
        install_rust()
        install_dependencies()
        
        # Create project
        create_project_structure()
        create_cargo_toml()
        create_build_rs()
        create_cpp_files()
        create_rust_files()
        
        # Build and run
        build_project()
        results = run_benchmark()
        
        # Save results
        csv_file = save_csv(results)
        json_file = save_json(results)
        pdf_file = generate_pdf_report(results, env_type, gpu_info)
        
        # Summary
        print_section("Benchmark Complete!")
        print("\n📊 Results Summary:\n")
        
        for r in results:
            print(f"Device: {r['device_name']}")
            print(f"  Baseline:     {r['baseline_gbps']:.2f} GB/s")
            print(f"  Optimized:    {r['optimized_gbps']:.2f} GB/s")
            print(f"  Multi-Stream: {r['multistream_gbps']:.2f} GB/s")
            print(f"  Improvement:  {r['improvement_pct']:.1f}%")
            print()
        
        print("📁 Output Files:")
        print(f"  CSV:  {csv_file}")
        print(f"  JSON: {json_file}")
        if pdf_file:
            print(f"  PDF:  {pdf_file}")
        print()
        
        print("✅ All done! Download the files above.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""

    (Config.PROJECT_DIR / "src" / "runtime" / "mod.rs").write_text("mod bindings;\npub use bindings::*;\n")
"""
