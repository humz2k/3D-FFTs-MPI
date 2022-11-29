#define fftPrecision double

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../dfft/a2adfft.h"
#include "cuda_utils.h"

void populate_grid(fftPrecision* myGridCells, int nlocal){
    for (int i = 0; i < (nlocal*2); i+=2){
        myGridCells[i] = ((fftPrecision)rand()) / (fftPrecision)RAND_MAX;
        myGridCells[i+1] = 0;
    }
}

int main(int argc, char** argv){

    MPI_Init(NULL, NULL);

    int Ng = 8;
    int blockSize = 64;

    a2aDistribution dist(MPI_COMM_WORLD,Ng);
    a2adfft dfft(dist);

    fftPrecision* data = (fftPrecision*) malloc(dist.nlocal * sizeof(fftPrecision) * 2);
    fftPrecision* scratch = (fftPrecision*) malloc(dist.nlocal * sizeof(fftPrecision) * 2);

    fftPrecision** d_Buff1 = (fftPrecision**) malloc(sizeof(fftPrecision*));
    fftPrecision** d_Buff2 = (fftPrecision**) malloc(sizeof(fftPrecision*));

    initialize_cuda(d_Buff1,d_Buff2,dist.nlocal);

    char name[] = "proc0"; char c = dist.world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");

    populate_grid(data,dist.nlocal);

    fwrite(data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    dfft.make_plans(scratch,d_Buff1,d_Buff2);

    dfft.forward(data);

    fwrite(data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    free(data);
    free(scratch);
    finalize_cuda(d_Buff1,d_Buff2);
    free(d_Buff1);
    free(d_Buff2);
    MPI_Finalize();

}

