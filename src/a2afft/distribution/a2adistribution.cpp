#define saveDims
#define dimsOutputFile "dims"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include "kernels/kernels.h"
#include "a2adistribution.h"

/*
Calculates the coordinates of the rank from the dimensions from MPI_Dims_create.
*/
void rank2coords(int rank, int* dims, int* out){

  out[0] = rank / (dims[1]*dims[2]);
  out[1] = (rank - out[0]*(dims[1]*dims[2])) / dims[2];
  out[2] = rank - out[0]*(dims[1]*dims[2]) - out[1]*(dims[2]);

}

/*
Gets the dimensions of the local grid of particles from the dimensions from MPI_Dims_create.
*/
void topology2localgrid(int Ng, int* dims, int* grid_size){

    grid_size[0] = Ng / dims[0];
    grid_size[1] = Ng / dims[1];
    grid_size[2] = Ng / dims[2];

}

/*
Gets the global coordinates of the starting point of the local grid.
*/
void get_local_grid_start(int* local_grid_size, int* coords, int* local_coordinates_start){

    local_coordinates_start[0] = local_grid_size[0] * coords[0];
    local_coordinates_start[1] = local_grid_size[1] * coords[1];
    local_coordinates_start[2] = local_grid_size[2] * coords[2];

}

a2aDistribution::a2aDistribution(MPI_Comm input_comm, int input_Ng, int input_blockSize)

: dims {0,0,0}, coords {0,0,0}, local_grid_size {0,0,0}, local_coordinates_start {0,0,0}

{

    ndims = 3;
    Ng = input_Ng;
    comm = input_comm;

    blockSize = input_blockSize;

    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    MPI_Dims_create(world_size,ndims,dims);

    assert((Ng % dims[0]) == 0);
    assert((Ng % dims[1]) == 0);
    assert((Ng % dims[2]) == 0);

    rank2coords(world_rank,dims,coords);
    topology2localgrid(Ng,dims,local_grid_size);
    get_local_grid_start(local_grid_size,coords,local_coordinates_start);

    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    MPI_Type_contiguous((int)sizeof(fftPrecision) * 2, MPI_BYTE, &TYPE_COMPLEX);
    MPI_Type_commit(&TYPE_COMPLEX);

    assert((world_size % dims[0]) == 0);
    assert((world_size % dims[1]) == 0);
    assert((world_size % dims[2]) == 0);

    assert(((local_grid_size[0] * local_grid_size[1]) % world_size) == 0);
    assert(((local_grid_size[0] * local_grid_size[2]) % world_size) == 0);
    assert(((local_grid_size[1] * local_grid_size[2]) % world_size) == 0);

    #ifndef cudampi
    h_scratch1 = (fftPrecision*) malloc(nlocal * sizeof(fftPrecision) * 2);
    h_scratch2 = (fftPrecision*) malloc(nlocal * sizeof(fftPrecision) * 2);
    #endif

    #ifdef verbose    
    if (world_rank == 0){
        printf("#######\nDISTRIBUTION PARAMETERS\n");
        printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);
        printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("NLOCAL: %d\n",nlocal);
        printf("#######\n\n");
    }
    #endif

    #ifdef saveDims
    if (world_rank == 0){
        FILE *dimsfile = fopen(dimsOutputFile, "w");
        fwrite(&world_size,sizeof(int),1,dimsfile);
        fwrite(&Ng,sizeof(int),1,dimsfile);
        fwrite(&nlocal,sizeof(int),1,dimsfile);
        fwrite(dims,sizeof(int),3,dimsfile);
        fwrite(local_grid_size,sizeof(int),3,dimsfile);
        fclose(dimsfile);
    }
    #endif

}

void a2aDistribution::getZPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff1,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff2,h_scratch2,nlocal);
    #else
    MPI_Alltoall(d_Buff1[0],nsends,TYPE_COMPLEX,d_Buff2[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

    launch_d_z_a2a_to_z_pencils(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

}

void a2aDistribution::returnZPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){
    
    int nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    launch_d_z_pencils_to_z_a2a(d_Buff1, d_Buff2, blockSize, world_size, nlocal, local_grid_size, dims);

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff2,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff1,h_scratch2,nlocal);
    #else
    MPI_Alltoall(d_Buff2[0],nsends,TYPE_COMPLEX,d_Buff1[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

}

void a2aDistribution::getXPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    launch_d_fast_z_to_x(d_Buff1, d_Buff2, local_grid_size, blockSize, nlocal);

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff2,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff1, h_scratch2, nlocal);
    #else
    MPI_Alltoall(d_Buff2[0],nsends,TYPE_COMPLEX,d_Buff1[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

    launch_d_x_a2a_to_x_pencils(d_Buff1,d_Buff2,blockSize,world_size,nlocal,local_grid_size,dims);

}

void a2aDistribution::returnXPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    launch_d_x_pencils_to_x_a2a(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff1,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff2,h_scratch2,nlocal);
    #else
    MPI_Alltoall(d_Buff1[0],nsends,TYPE_COMPLEX,d_Buff2[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

}

void a2aDistribution::getYPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    launch_d_fast_x_to_y(d_Buff2, d_Buff1, local_grid_size, blockSize, nlocal);

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff1,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff2, h_scratch2, nlocal);
    #else
    MPI_Alltoall(d_Buff1[0],nsends,TYPE_COMPLEX,d_Buff2[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

    launch_d_y_a2a_to_y_pencils(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

}

void a2aDistribution::returnYPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    launch_d_y_pencils_to_y_a2a(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

    #ifndef cudampi
    copy_d2h(h_scratch1,d_Buff1,nlocal);
    MPI_Alltoall(h_scratch1,nsends,TYPE_COMPLEX,h_scratch2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    copy_h2d(d_Buff2, h_scratch2, nlocal);
    #else
    MPI_Alltoall(d_Buff1[0],nsends,TYPE_COMPLEX,d_Buff2[0],nsends,TYPE_COMPLEX,MPI_COMM_WORLD);
    #endif

    launch_d_fast_y_to_z(d_Buff2, d_Buff1, local_grid_size, blockSize, nlocal);

}

void a2aDistribution::finalize(){
    MPI_Type_free(&TYPE_COMPLEX);
    
    #ifndef cudampi
    free(h_scratch1);
    free(h_scratch2);
    #endif
}