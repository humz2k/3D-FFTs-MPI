#define fftPrecision double

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
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

    assert(((argc-1)%2) == 0);

    int Ng = 8;
    int blockSize = 64;

    for (int i = 1; i < argc; i += 2){
        if (strcmp(argv[i],"-Ng") == 0){
            Ng = atoi(argv[i+1]);
        }
        if (strcmp(argv[i],"-blockSize") == 0){
            blockSize = atoi(argv[i+1]);
        }
    }

    a2aDistribution dist(MPI_COMM_WORLD,Ng,blockSize);
    a2adfft dfft(dist);

    int world_rank = dist.world_rank;
    int nlocal = dist.nlocal;

    if (world_rank == 0){
        printf("#######\nINPUT PARAMETERS\nNg = %d\nblockSize = %d\n#######\n\n",Ng,blockSize);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Malloc arrays...\n");}

    fftPrecision* data = (fftPrecision*) malloc(nlocal * sizeof(fftPrecision) * 2);
    fftPrecision* scratch = (fftPrecision*) malloc(nlocal * sizeof(fftPrecision) * 2);

    fftPrecision** d_Buff1 = (fftPrecision**) malloc(sizeof(fftPrecision*));
    fftPrecision** d_Buff2 = (fftPrecision**) malloc(sizeof(fftPrecision*));

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished Malloc arrays\nInitializing cuda...\n");}

    initialize_cuda(d_Buff1,d_Buff2,nlocal);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished initializing cuda\nCreating output files...");}

    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished creating output files\nPopulating grid...");}

    populate_grid(data,nlocal);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished populating grid\nWriting initial data to file...");}

    fwrite(data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing initial data to file\nMaking dfft plans...");}

    dfft.make_plans(scratch,d_Buff1,d_Buff2);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished making dfft plans\nCalling dfft.forward()...\n\n");}

    dfft.forward(data);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("\nFinished dfft.forward()\nWriting transformed data to file...\n");}

    fwrite(data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing transformed data to file\nCalling dfft.inverse()...\n\n");}

    dfft.inverse(data);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("\nFinished dfft.inverse()\nWriting inverse transformed data to file...\n");}

    fwrite(data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing inverse transformed data to file\nFinalizing...\n");}

    fclose(out_file);
    free(data);
    free(scratch);
    finalize_cuda(d_Buff1,d_Buff2);
    free(d_Buff1);
    free(d_Buff2);
    dist.finalize();
    MPI_Finalize();

}

