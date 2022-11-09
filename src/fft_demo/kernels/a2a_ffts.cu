#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


extern "C" {
    void initialize_cuda(float* d_myGridCellsBuff1, float* d_myGridCellsBuff2, int nlocal){
        cudaFree(0);
        cudaMalloc(&d_myGridCellsBuff1,nlocal * 2 * sizeof(float));
        cudaMalloc(&d_myGridCellsBuff2,nlocal * 2 * sizeof(float));
    }
}

extern "C" {
    void copy_h2d(float* source, float* dest, int nlocal){
        cudaMemcpy(dest,source,nlocal * 2 * sizeof(float),cudaMemcpyHostToDevice);

        float* test = (float*) malloc(nlocal * sizeof(float) * 2);

        cudaMemcpy(test,dest,nlocal*sizeof(float)*2,cudaMemcpyDeviceToHost);

        printf("TEST %f\n",test[2]);

        free(test);

    }
}

extern "C" {
    void copy_d2h(float* source, float* dest, int nlocal){
        cudaMemcpy(dest,source,nlocal * 2 * sizeof(float),cudaMemcpyDeviceToHost);
    }
}

extern "C" {
    void finalize_cuda(float* d_myGridCellsBuff1, float* d_myGridCellsBuff2){
        cudaFree(d_myGridCellsBuff1);
        cudaFree(d_myGridCellsBuff2);
    }
}