#define showDims
#define saveDims
#define dimsOutputFile "dims.out"

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

a2aDistribution::a2aDistribution(MPI_Comm input_comm, int input_Ng){

    ndims = 3;
    Ng = input_Ng;
    comm = input_comm;

    int world_size; MPI_Comm_size(comm, &world_size);
    int world_rank; MPI_Comm_rank(comm, &world_rank);

    int dims[3] = {0,0,0}; MPI_Dims_create(world_size,ndims,dims);

    assert((Ng % dims[0]) == 0);
    assert((Ng % dims[1]) == 0);
    assert((Ng % dims[2]) == 0);

    int coords[3] = {0,0,0}; rank2coords(world_rank,dims,coords);
    int local_grid_size[3] = {0,0,0}; topology2localgrid(Ng,dims,local_grid_size);
    int local_coordinates_start[3] = {0,0,0}; get_local_grid_start(local_grid_size,coords,local_coordinates_start);

    int nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    MPI_Type_contiguous((int)sizeof(fftPrecision) * 2, MPI_BYTE, &TYPE_COMPLEX);
    MPI_Type_commit(&TYPE_COMPLEX);

    assert((world_size % dims[0]) == 0);
    assert((world_size % dims[1]) == 0);
    assert((world_size % dims[2]) == 0);

    assert(((local_grid_size[0] * local_grid_size[1]) % world_size) == 0);
    assert(((local_grid_size[0] * local_grid_size[2]) % world_size) == 0);
    assert(((local_grid_size[1] * local_grid_size[2]) % world_size) == 0);

    #ifdef showDims      
    if (world_rank == 0){
        printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);
        printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("NLOCAL: %d\n",nlocal);
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

void a2aDistribution::getZPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(data,nsends,TYPE_COMPLEX,scratch,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_Buff2,scratch,nlocal);

    launch_d_z_a2a_to_z_pencils(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

}

void a2aDistribution::returnZPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){
    
    int nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    launch_d_z_pencils_to_z_a2a(d_Buff1, d_Buff2, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(scratch,d_Buff2,nlocal);

    MPI_Alltoall(scratch,nsends,TYPE_COMPLEX,data,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

}

void a2aDistribution::getXPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    copy_h2d(d_Buff2, data, nlocal);

    launch_d_fast_z_to_x(d_Buff2, d_Buff1, local_grid_size, blockSize, nlocal);

    copy_d2h(data,d_Buff1,nlocal);

    MPI_Alltoall(data,nsends,TYPE_COMPLEX,scratch,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_Buff2, scratch, nlocal);

    launch_d_x_a2a_to_x_pencils(d_Buff2,d_Buff1,blockSize,world_size,nlocal,local_grid_size,dims);

}

void a2aDistribution::returnXPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    launch_d_x_pencils_to_x_a2a(d_Buff1, d_Buff2, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(scratch,d_Buff2,nlocal);

    MPI_Alltoall(scratch,nsends,TYPE_COMPLEX,data,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

}

void a2aDistribution::getYPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    copy_h2d(d_Buff2, data, nlocal);

    launch_d_fast_x_to_y(d_Buff2, d_Buff1, local_grid_size, blockSize, nlocal);

    copy_d2h(data,d_Buff1,nlocal);

    MPI_Alltoall(data,nsends,TYPE_COMPLEX,scratch,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_Buff2, scratch, nlocal);

    launch_d_y_a2a_to_y_pencils(d_Buff2, d_Buff1, blockSize, world_size, nlocal, local_grid_size, dims);

}

void a2aDistribution::returnYPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2){

    int nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    launch_d_y_pencils_to_y_a2a(d_Buff1, d_Buff2, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(data,d_Buff2,nlocal);

    MPI_Alltoall(data,nsends,TYPE_COMPLEX,scratch,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_Buff2, scratch, nlocal);

    launch_d_fast_y_to_z(d_Buff2, d_Buff1, local_grid_size, blockSize, nlocal);

    copy_d2h(data,d_Buff1,nlocal);

}