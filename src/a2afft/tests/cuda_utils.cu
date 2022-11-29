#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_utils.h"

extern void initialize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2, int nlocal){
    cudaFree(0);
    if (cudaMalloc(&(d_myGridCellsBuff1[0]),nlocal * 2 * sizeof(fftPrecision)) != cudaSuccess){
        printf("MALLOC ERROR >> %s\n", cudaGetErrorString(cudaGetLastError()));
    };
    if (cudaMalloc(&(d_myGridCellsBuff2[0]),nlocal * 2 * sizeof(fftPrecision)) != cudaSuccess){
        printf("MALLOC ERROR >> %s\n", cudaGetErrorString(cudaGetLastError()));
    };
}

extern void finalize_cuda(fftPrecision** d_myGridCellsBuff1, fftPrecision** d_myGridCellsBuff2){
    if (cudaFree(d_myGridCellsBuff1[0]) != cudaSuccess){
        printf("cudaFree Error >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaFree(d_myGridCellsBuff2[0]) != cudaSuccess){
        printf("cudaFree Error >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
}