mod memory;
use std::time::Instant;

extern "C" {
    fn cuda_malloc(size: usize) -> *mut std::ffi::c_void;
    fn cuda_register_host(ptr: *mut std::ffi::c_void, size: usize) -> i32;
    fn cuda_stream_create() -> *mut std::ffi::c_void;
    fn cuda_stream_sync(stream: *mut std::ffi::c_void);
    fn cuda_memcpy_async_h2d(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, size: usize, stream: *mut std::ffi::c_void) -> i32;
}

fn main() {
    let host_size = 2 * 1024 * 1024 * 1024; // 2GB
    let transfer_size = 256 * 1024 * 1024; // 256MB Chunks
    
    unsafe {
        let host_ptr = memory::allocate_huge_pages(host_size);
        let gpu_ptr = cuda_malloc(transfer_size);
        let stream = cuda_stream_create();

        // Lockdown memory (Pinning)
        cuda_register_host(host_ptr as *mut _, host_size);

        println!("🚀 L3-Stage Engine: Starting Benchmark...");
        let start = Instant::now();

        for _ in 0..40 {
            cuda_memcpy_async_h2d(gpu_ptr, host_ptr as *const _, transfer_size, stream);
            cuda_stream_sync(stream);
        }

        let elapsed = start.elapsed().as_secs_f64();
        let gb_total = (transfer_size as f64 * 40.0) / 1e9;
        println!("✅ Throughput: {:.2} GB/s", gb_total / elapsed);
    }
}
