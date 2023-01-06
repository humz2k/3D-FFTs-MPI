#define fftPrecision double
#define timing

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../a2a_3D_FFTs.h"
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

    #ifdef verbose
    if (world_rank == 0){
        printf("#######\nINPUT PARAMETERS\nNg = %d\nblockSize = %d\n#######\n\n",Ng,blockSize);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Malloc arrays...\n");}
    #endif

    fftPrecision* temp_data = (fftPrecision*) malloc(nlocal * sizeof(fftPrecision) * 2);

    fftPrecision** d_data = (fftPrecision**) malloc(sizeof(fftPrecision*));
    fftPrecision** d_scratch = (fftPrecision**) malloc(sizeof(fftPrecision*));

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished Malloc arrays\nInitializing cuda...\n");}
    #endif

    initialize_cuda(d_data,d_scratch,nlocal);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished initializing cuda\nCreating output files...");}
    #endif

    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished creating output files\nPopulating grid...");}
    #endif

    populate_grid(temp_data,nlocal);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished populating grid\nWriting initial data to file...");}
    #endif

    fwrite(temp_data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing initial data to file\nMaking dfft plans...");}
    #endif

    dfft.make_plans(d_scratch);

    util_copy_h2d(d_data,temp_data,nlocal);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished making dfft plans\nCalling dfft.forward()...\n\n");}
    #endif

    dfft.forward(d_data);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("\nFinished dfft.forward()\nWriting transformed data to file...\n");}
    #endif
    
    util_copy_d2h(temp_data,d_data,nlocal);

    fwrite(temp_data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing transformed data to file\nCalling dfft.inverse()...\n\n");}
    #endif

    dfft.inverse(d_data);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("\nFinished dfft.inverse()\nWriting inverse transformed data to file...\n");}
    #endif

    util_copy_d2h(temp_data,d_data,nlocal);

    fwrite(temp_data,sizeof(fftPrecision),dist.nlocal*2,out_file);

    #ifdef verbose
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0){printf("Finished writing inverse transformed data to file\nFinalizing...\n");}
    #endif
    

    fclose(out_file);
    free(temp_data);
    
    finalize_cuda(d_data,d_scratch);
    free(d_data);
    free(d_scratch);

    dist.finalize();
    MPI_Finalize();

}

