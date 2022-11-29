#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernels.h"

extern void copy_h2d(fftPrecision** dest, fftPrecision* source, int nlocal){
    if (cudaMemcpy(dest[0],source,nlocal * 2 * sizeof(fftPrecision),cudaMemcpyHostToDevice) != cudaSuccess){
        printf("Memcpy Error h2d >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
}

extern void copy_d2h(fftPrecision* dest, fftPrecision** source, int nlocal){
    if (cudaMemcpy(dest,source[0],nlocal * 2 * sizeof(fftPrecision),cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("Memcpy Error d2h >> %s\n",cudaGetErrorString(cudaGetLastError()));
    }
}