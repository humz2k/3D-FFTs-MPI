#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"

extern void forward_1d_fft(fftPrecision** data, int Ng, int nlocal){

    cufftHandle plan;

    int nFFTs = nlocal / Ng;

    #ifdef doublePrecision

    if (cufftPlan1d(&plan, Ng, CUFFT_Z2Z, nFFTs) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }

    if (cufftExecZ2Z(plan, (cudafftPrecision*)data[0], (cudafftPrecision*)data[0], CUFFT_FORWARD) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }

    #else

    if (cufftPlan1d(&plan, Ng, CUFFT_C2C, nFFTs) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }

    if (cufftExecC2C(plan, (cudafftPrecision*)data[0], (cudafftPrecision*)data[0], CUFFT_FORWARD) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C Forward failed\n");
        return;	
    }
    #endif

    cudaDeviceSynchronize();

}

extern void inverse_1d_fft(fftPrecision** data, int Ng, int nlocal){

    cufftHandle plan;

    int nFFTs = nlocal / Ng;

    #ifdef doublePrecision

    if (cufftPlan1d(&plan, Ng, CUFFT_Z2Z, nFFTs) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }

    if (cufftExecZ2Z(plan, (cudafftPrecision*)data[0], (cudafftPrecision*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z Forward failed\n");
        return;	
    }

    #else

    if (cufftPlan1d(&plan, Ng, CUFFT_C2C, nFFTs) != CUFFT_SUCCESS){
        printf("CUFFT error: Plan creation failed\n");
        return;	
    }

    if (cufftExecC2C(plan, (cudafftPrecision*)data[0], (cudafftPrecision*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C Forward failed\n");
        return;	
    }

    #endif

    cudaDeviceSynchronize();

    }

__global__
void scale_fft(fftPrecision* __restrict out, const fftPrecision* __restrict data, int Ng, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    fftPrecision scale = (fftPrecision)(Ng * Ng * Ng);

    if (idx < nlocal){

        out[idx*2] = data[idx*2] / scale;
        out[idx*2 + 1] = data[idx*2 + 1] / scale;

    }

}

__global__
void fast_copy_fft(fftPrecision* __restrict out, const fftPrecision* __restrict data, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        out[idx*2] = data[idx*2];
        out[idx*2 + 1] = data[idx*2 + 1];

    }

}

extern void launch_scale_fft(fftPrecision** out, fftPrecision** data, int Ng, int nlocal, int blockSize){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    scale_fft<<<numBlocks,blockSize>>>(out[0],data[0],Ng,nlocal);

    cudaDeviceSynchronize();

}

extern void launch_fast_copy_fft(fftPrecision** out, fftPrecision** data, int nlocal, int blockSize){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    fast_copy_fft<<<numBlocks,blockSize>>>(out[0],data[0],nlocal);

    cudaDeviceSynchronize();

}