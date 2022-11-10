#include <cufft.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

    void forward_1d_fft(float** data, int Ng, int nlocal){

        cufftHandle plan;

        int nFFTs = nlocal / Ng;

        if (cufftPlan1d(&plan, Ng, CUFFT_C2C, nFFTs) != CUFFT_SUCCESS){
            printf("CUFFT error: Plan creation failed\n");
            return;	
        }

        if (cufftExecC2C(plan, (cufftComplex*)data[0], (cufftComplex*)data[0], CUFFT_FORWARD) != CUFFT_SUCCESS){
            printf("CUFFT error: ExecC2C Forward failed\n");
            return;	
        }

        cudaDeviceSynchronize();

        /*if (cufftExecC2C(plan, (cufftComplex*)data[0], (cufftComplex*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
            printf("CUFFT error: ExecC2C Forward failed\n");
            return;	
        }*/

    }

}

extern "C" {

    void inverse_1d_fft(float** data, int Ng, int nlocal){

        cufftHandle plan;

        int nFFTs = nlocal / Ng;

        if (cufftPlan1d(&plan, Ng, CUFFT_C2C, nFFTs) != CUFFT_SUCCESS){
            printf("CUFFT error: Plan creation failed\n");
            return;	
        }

        if (cufftExecC2C(plan, (cufftComplex*)data[0], (cufftComplex*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
            printf("CUFFT error: ExecC2C Forward failed\n");
            return;	
        }

        cudaDeviceSynchronize();

        /*if (cufftExecC2C(plan, (cufftComplex*)data[0], (cufftComplex*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
            printf("CUFFT error: ExecC2C Forward failed\n");
            return;	
        }*/

    }

}

__global__
void scale_fft(float* data, int Ng, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float scale = (float)(Ng * Ng * Ng);

    if (idx < nlocal){

        data[idx*2] = data[idx*2] / scale;
        data[idx*2 + 1] = data[idx*2 + 1] / scale;

    }

}

extern "C" {

    void launch_scale_fft(float** data, int Ng, int nlocal, int blockSize){

        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        scale_fft<<<numBlocks,blockSize>>>(data[0],Ng,nlocal);

        cudaDeviceSynchronize();

    }

}