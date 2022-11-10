#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../kernels/kernels.h"

extern void forward_fft3d(float* myGridCellsBuff1, float* myGridCellsBuff2, float** d_myGridCellsBuff1, float** d_myGridCellsBuff2, int Ng, int nlocal, int world_size, int* local_grid_size, int* dims, int blockSize, MPI_Datatype TYPE_COMPLEX){

    int nsends;

    ////////////////////////////////////
    //        3D FFT Algorithm        // 
    ////////////////////////////////////

    //////////
    //Z AXIS//
    //////////

    nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1,myGridCellsBuff2,nlocal);

    launch_d_z_a2a_to_z_pencils(d_myGridCellsBuff1, d_myGridCellsBuff2, blockSize, world_size, nlocal, local_grid_size, dims);

    forward_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_z_pencils_to_z_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    //////////
    //X AXIS//
    ////////// 

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_z_to_x(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

    nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_x_a2a_to_x_pencils(d_myGridCellsBuff1,d_myGridCellsBuff2,blockSize,world_size,nlocal,local_grid_size,dims);

    forward_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_x_pencils_to_x_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    //////////
    //Y AXIS//
    ////////// 

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_x_to_y(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);
    
    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

    nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_y_a2a_to_y_pencils(d_myGridCellsBuff1, d_myGridCellsBuff2, blockSize, world_size, nlocal, local_grid_size, dims);

    forward_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_y_pencils_to_y_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    ////////////
    //Finalize//
    ////////////  

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_y_to_z(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

}

extern void inverse_fft3d(float* myGridCellsBuff1, float* myGridCellsBuff2, float** d_myGridCellsBuff1, float** d_myGridCellsBuff2, int Ng, int nlocal, int world_size, int* local_grid_size, int* dims, int blockSize, MPI_Datatype TYPE_COMPLEX){

    int nsends;

    ////////////////////////////////////
    //        3D FFT Algorithm        // 
    ////////////////////////////////////

    //////////
    //Z AXIS//
    //////////

    nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1,myGridCellsBuff2,nlocal);

    launch_d_z_a2a_to_z_pencils(d_myGridCellsBuff1, d_myGridCellsBuff2, blockSize, world_size, nlocal, local_grid_size, dims);

    inverse_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_z_pencils_to_z_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    //////////
    //X AXIS//
    ////////// 

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_z_to_x(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

    nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_x_a2a_to_x_pencils(d_myGridCellsBuff1,d_myGridCellsBuff2,blockSize,world_size,nlocal,local_grid_size,dims);

    inverse_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_x_pencils_to_x_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    //////////
    //Y AXIS//
    ////////// 

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_x_to_y(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);
    
    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

    nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_y_a2a_to_y_pencils(d_myGridCellsBuff1, d_myGridCellsBuff2, blockSize, world_size, nlocal, local_grid_size, dims);

    inverse_1d_fft(d_myGridCellsBuff2, Ng, nlocal);

    launch_d_y_pencils_to_y_a2a(d_myGridCellsBuff2, d_myGridCellsBuff1, blockSize, world_size, nlocal, local_grid_size, dims);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff1,nlocal);

    MPI_Alltoall(myGridCellsBuff1,nsends,TYPE_COMPLEX,myGridCellsBuff2,nsends,TYPE_COMPLEX,MPI_COMM_WORLD);

    ////////////
    //Finalize//
    ////////////  

    copy_h2d(d_myGridCellsBuff1, myGridCellsBuff2,nlocal);

    launch_d_fast_y_to_z(d_myGridCellsBuff1, d_myGridCellsBuff2, local_grid_size, blockSize, nlocal);

    launch_scale_fft(d_myGridCellsBuff2, Ng, nlocal, blockSize);

    copy_d2h(myGridCellsBuff1,d_myGridCellsBuff2,nlocal);

}