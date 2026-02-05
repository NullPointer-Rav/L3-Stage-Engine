# L3-Stage Engine v2.0 🚀

A high-performance PCIe DMA orchestrator written in Rust and CUDA. This engine bypasses standard OS memory bottlenecks to achieve near-theoretical throughput limits.

## 📊 Performance Benchmark
- **Standard Baseline:** 4.43 GB/s
- **L3-Stage Engine:** **12.25 GB/s** (+168% improvement)



## 🛠 Features
- **Zero-Copy Architecture:** Uses `cudaHostRegister` for direct DMA hardware access.
- **Huge Page Backing:** Reduces TLB misses via `MAP_HUGETLB`.
- **Async Pipelining:** Overlaps data movement with dual-stream orchestration.

## 🚀 Getting Started
1. Ensure CUDA Toolkit is installed.
2. Clone the repo: `git clone https://github.com/NullPointer-Rav/L3-Stage-Engine`
3. Build: `cargo build --release`
4. Run: `./target/release/l3_stage_engine`

## 📜 Citation
If you use this work in your research, please cite:
> Mohamed Sayed. (2026). *L3-Stage Engine: Optimizing PCIe Throughput via Async Pinned Memory Pipelining.*
