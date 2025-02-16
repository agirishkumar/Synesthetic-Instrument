// include/cuda_utils.h
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if(err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1); \
        } \
    } while(0)

// Helper function to calculate grid dimensions
inline dim3 calculate_grid_dim(int num_elements, int block_size) {
    return dim3((num_elements + block_size - 1) / block_size);
}