use std::ptr;
use libc::{mmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE, MAP_HUGETLB};

/// Allocates memory using Huge Pages (2MB/1GB) for TLB efficiency.
/// Falls back to standard 4KB pages if Huge Pages are unavailable.
pub unsafe fn allocate_huge_pages(size: usize) -> *mut u8 {
    let mut ptr = mmap(ptr::null_mut(), size, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    
    if ptr == libc::MAP_FAILED {
        ptr = mmap(ptr::null_mut(), size, PROT_READ | PROT_WRITE, 
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    ptr as *mut u8
}
