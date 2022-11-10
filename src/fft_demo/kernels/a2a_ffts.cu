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

        /*if (cufftExecC2C(plan, (cufftComplex*)data[0], (cufftComplex*)data[0], CUFFT_INVERSE) != CUFFT_SUCCESS){
            printf("CUFFT error: ExecC2C Forward failed\n");
            return;	
        }*/

    }

}