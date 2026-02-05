fn main() {
    // Dynamically link CUDA libraries
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    
    // Compile the C++/CUDA wrapper
    cc::Build::new()
        .cpp(true)
        .file("src/cuda_wrapper.cpp")
        .flag("-I/usr/local/cuda/include")
        .compile("cuda_wrapper");
}
