#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "kernels/kernels.h"
#include "fft_3d/fft_3d.h"
#include "initializer/initializer.h"

int main(int argc, char** argv){

    MPI_Init(NULL, NULL);

    assert(argc == 2);

    int ndims = 3;
    int Ng = atoi(argv[1]);
    int blockSize = 64;

    int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int dims[3] = {0,0,0}; MPI_Dims_create(world_size,ndims,dims);
    int coords[3] = {0,0,0}; rank2coords(world_rank,dims,coords);
    int local_grid_size[3] = {0,0,0}; topology2localgrid(Ng,dims,local_grid_size);
    int local_coordinates_start[3] = {0,0,0}; get_local_grid_start(local_grid_size,coords,local_coordinates_start);
    int nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    assert((world_size % dims[0]) == 0);
    assert((world_size % dims[1]) == 0);
    assert((world_size % dims[2]) == 0);
    assert(((local_grid_size[0] * local_grid_size[1]) % world_size) == 0);
    assert(((local_grid_size[0] * local_grid_size[2]) % world_size) == 0);
    assert(((local_grid_size[1] * local_grid_size[2]) % world_size) == 0);

    if (world_rank == 0){
        printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);
        printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("NLOCAL: %d\n",nlocal);

        FILE *params = fopen("params", "w");

        fwrite(&world_size,sizeof(int),1,params);
        fwrite(&Ng,sizeof(int),1,params);
        fwrite(&nlocal,sizeof(int),1,params);
        fwrite(dims,sizeof(int),3,params);
        fwrite(local_grid_size,sizeof(int),3,params);

        fclose(params);

    }

    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");

    float* myGridCellsBuff1 = (float*) malloc(nlocal * sizeof(float) * 2);
    float* myGridCellsBuff2 = (float*) malloc(nlocal * sizeof(float) * 2);

    float** d_myGridCellsBuff1 = (float**) malloc(sizeof(float*));
    float** d_myGridCellsBuff2 = (float**) malloc(sizeof(float*));

    initialize_cuda(d_myGridCellsBuff1, d_myGridCellsBuff2, nlocal);

    //int nsends;

    MPI_Datatype TYPE_COMPLEX;
    MPI_Type_contiguous((int)sizeof(float) * 2, MPI_BYTE, &TYPE_COMPLEX);
    MPI_Type_commit(&TYPE_COMPLEX);

    populate_grid(myGridCellsBuff1,nlocal);
    //populate_grid(myGridCellsBuff1, nlocal, world_rank, local_coordinates_start, local_grid_size, Ng);

    if (world_rank == 0){
        printf("rank %d myGridCells:\n  ",world_rank);
        int nprint = 5;
        for (int i = 0; i < (nprint-1)*2; i+=2){
            printf("%f + %fi, ",myGridCellsBuff1[i],myGridCellsBuff1[i+1]);
        }
        printf("%f + %fi ...\n",myGridCellsBuff1[(nprint-1)*2],myGridCellsBuff1[(nprint-1)*2 + 1]);
    }

    fwrite(myGridCellsBuff1,sizeof(float),nlocal*2,out_file);

    forward_fft3d(myGridCellsBuff1, myGridCellsBuff2, d_myGridCellsBuff1, d_myGridCellsBuff2, Ng, nlocal, world_size, local_grid_size, dims, blockSize, TYPE_COMPLEX);

    fwrite(myGridCellsBuff1,sizeof(float),nlocal*2,out_file);

    inverse_fft3d(myGridCellsBuff1, myGridCellsBuff2, d_myGridCellsBuff1, d_myGridCellsBuff2, Ng, nlocal, world_size, local_grid_size, dims, blockSize, TYPE_COMPLEX);

    fwrite(myGridCellsBuff1,sizeof(float),nlocal*2,out_file);

    fclose(out_file);
    
    free(myGridCellsBuff1);
    free(myGridCellsBuff2);
    finalize_cuda(d_myGridCellsBuff1,d_myGridCellsBuff2);
    free(d_myGridCellsBuff1);
    free(d_myGridCellsBuff2);
    MPI_Finalize();


}