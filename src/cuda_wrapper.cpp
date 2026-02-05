#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    void* cuda_malloc(size_t size) {
        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void cuda_free(void* ptr) {
        if (ptr) cudaFree(ptr);
    }

    int cuda_register_host(void* ptr, size_t size) {
        return (int)cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    }

    int cuda_unregister_host(void* ptr) {
        return (int)cudaHostUnregister(ptr);
    }

    void* cuda_stream_create() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return (void*)stream;
    }

    void cuda_stream_sync(void* stream) {
        cudaStreamSynchronize((cudaStream_t)stream);
    }

    int cuda_memcpy_async_h2d(void* dst, const void* src, size_t size, void* stream) {
        return (int)cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    }
}
