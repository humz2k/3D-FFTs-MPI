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

extern void util_copy_h2d(fftPrecision** dest, fftPrecision* source, int nlocal){
    if (cudaMemcpy(dest[0],source,nlocal * 2 * sizeof(fftPrecision),cudaMemcpyHostToDevice) != cudaSuccess){
        printf("Memcpy Error h2d >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
}

extern void util_copy_d2h(fftPrecision* dest, fftPrecision** source, int nlocal){
    if (cudaMemcpy(dest,source[0],nlocal * 2 * sizeof(fftPrecision),cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Memcpy Error d2h >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
}