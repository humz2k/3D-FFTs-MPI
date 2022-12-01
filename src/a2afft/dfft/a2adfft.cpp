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

void a2adfft::make_plans(fftPrecision* input_scratch, fftPrecision** input_d_Buff1, fftPrecision** input_d_Buff2){

    scratch = input_scratch;
    d_Buff1 = input_d_Buff1;
    d_Buff2 = input_d_Buff2;

    plans_made = 1;

}

void a2adfft::forward(fftPrecision* data){
    #ifdef verbose
    if (distribution.world_rank == 0){printf("Starting 3D FFT\nGetting Z Pencils...\n");}
    #endif

    distribution.getZPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Z Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nReturning Z Pencils...\n");}
    #endif

    distribution.returnZPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Z Pencils\nGetting X Pencils...\n");}
    #endif

    distribution.getXPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got X Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nReturning X Pencils...\n");}
    #endif

    distribution.returnXPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned X Pencils\nGetting Y Pencils...\n");}
    #endif

    distribution.getYPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Y Pencils\nStarting 1D FFT...\n");}
    #endif

    forward_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D FFT\nReturning Y Pencils...\n");}
    #endif

    distribution.returnYPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Y Pencils\nFinished 3D FFT\n");}
    #endif

}

void a2adfft::inverse(fftPrecision* data){

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Starting 3D iFFT\nGetting Z Pencils...\n");}
    #endif

    distribution.getZPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Z Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nReturning Z Pencils...\n");}
    #endif

    distribution.returnZPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Z Pencils\nGetting X Pencils...\n");}
    #endif

    distribution.getXPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got X Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nReturning X Pencils...\n");}
    #endif

    distribution.returnXPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned X Pencils\nGetting Y Pencils...\n");}
    #endif

    distribution.getYPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Got Y Pencils\nStarting 1D iFFT...\n");}
    #endif

    inverse_1d_fft(d_Buff1, Ng, nlocal);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 1D iFFT\nLaunching Scale iFFT...\n");}
    #endif

    launch_scale_fft(d_Buff1, Ng, nlocal, blockSize);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished Scale iFFT\nReturning Y Pencils...\n");}
    #endif

    distribution.returnYPencils(data,scratch,d_Buff1,d_Buff2);

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Returned Y Pencils\nFinished 3D iFFT\n");}
    #endif

}