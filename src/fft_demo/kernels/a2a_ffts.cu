#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


extern "C" {
    void initialize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2, int nlocal, float* myGridCellsBuff1){
        cudaFree(0);
        if (cudaMalloc(&(d_myGridCellsBuff1[0]),nlocal * 2 * sizeof(float)) != cudaSuccess){
            printf("MALLOC ERROR >> %s\n", cudaGetErrorString(cudaGetLastError()));
        };
        if (cudaMalloc(&(d_myGridCellsBuff2[0]),nlocal * 2 * sizeof(float)) != cudaSuccess){
            printf("MALLOC ERROR >> %s\n", cudaGetErrorString(cudaGetLastError()));
        };
    }
}

extern "C" {
    void copy_h2d(float** dest, float* source, int nlocal){
        if (cudaMemcpy(dest[0],source,nlocal * 2 * sizeof(float),cudaMemcpyHostToDevice) != cudaSuccess){
            printf("Memcpy Error h2d >> %s\n",cudaGetErrorString(cudaGetLastError()));
        }

    }
}

extern "C" {
    void copy_d2h(float* dest, float** source, int nlocal){
        if (cudaMemcpy(dest,source[0],nlocal * 2 * sizeof(float),cudaMemcpyDeviceToHost) != cudaSuccess){
            printf("Memcpy Error d2h >> %s\n",cudaGetErrorString(cudaGetLastError()));
        }
    }
}

extern "C" {
    void finalize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2){
        if (cudaFree(d_myGridCellsBuff1[0]) != cudaSuccess){
            printf("cudaFree Error >> %s\n",cudaGetErrorString(cudaGetLastError()));
        }
        if (cudaFree(d_myGridCellsBuff2[0]) != cudaSuccess){
            printf("cudaFree Error >> %s\n",cudaGetErrorString(cudaGetLastError()));
        }
    }
}