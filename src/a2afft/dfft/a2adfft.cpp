#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "kernels/kernels.h"
#include "a2adfft.h"

a2adfft::a2adfft(a2aDistribution &dist) : distribution(dist){

    Ng = distribution.Ng;
    world_size = distribution.world_size;
    nlocal = distribution.nlocal;
    blockSize = distribution.blockSize;

    plans_made = 0;

}

void a2adfft::make_plans(fftPrecision** input_d_scratch){

    d_scratch = input_d_scratch;

    plans_made = 1;

}

void a2adfft::forward(fftPrecision** d_data){
    #ifdef verbose
    if (distribution.world_rank == 0){printf("Starting 3D FFT\nGetting Z Pencils...\n");}
    #endif

    distribution.getZPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Z Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_data, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nReturning Z Pencils...\n");}
    #endif

    distribution.returnZPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Z Pencils\nGetting X Pencils...\n");}
    #endif

    distribution.getXPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got X Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_scratch, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nReturning X Pencils...\n");}
    #endif

    distribution.returnXPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned X Pencils\nGetting Y Pencils...\n");}
    #endif

    distribution.getYPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Y Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_data, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nCopying back to input array...\n");}
    #endif

    launch_fast_copy_fft(d_scratch, d_data, nlocal, blockSize);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished copying back to input array\nReturning Y Pencils...\n");}
    #endif

    distribution.returnYPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Y Pencils\nFinished 3D FFT\n");}
    #endif

}

void a2adfft::inverse(fftPrecision** d_data){

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Starting 3D iFFT\nGetting Z Pencils...\n");}
    #endif

    distribution.getZPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Z Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_data, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nReturning Z Pencils...\n");}
    #endif

    distribution.returnZPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Z Pencils\nGetting X Pencils...\n");}
    #endif

    distribution.getXPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got X Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_scratch, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nReturning X Pencils...\n");}
    #endif

    distribution.returnXPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned X Pencils\nGetting Y Pencils...\n");}
    #endif

    distribution.getYPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Y Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_data, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nLaunching Scale iFFT...\n");}
    #endif

    launch_scale_fft(d_scratch, d_data, Ng, nlocal, blockSize);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished Scale iFFT\nReturning Y Pencils...\n");}
    #endif

    distribution.returnYPencils(d_data,d_scratch);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Y Pencils\nFinished 3D iFFT\n");}
    #endif

}