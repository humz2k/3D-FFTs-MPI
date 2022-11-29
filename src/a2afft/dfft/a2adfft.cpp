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

    distribution.getZPencils(data,scratch,d_Buff1,d_Buff2);
    forward_1d_fft(d_Buff1, Ng, nlocal);
    distribution.returnZPencils(data,scratch,d_Buff1,d_Buff2);

    distribution.getXPencils(data,scratch,d_Buff1,d_Buff2);
    forward_1d_fft(d_Buff1, Ng, nlocal);
    distribution.returnXPencils(data,scratch,d_Buff1,d_Buff2);

    distribution.getYPencils(data,scratch,d_Buff1,d_Buff2);
    forward_1d_fft(d_Buff1, Ng, nlocal);
    distribution.returnYPencils(data,scratch,d_Buff1,d_Buff2);

}

void a2adfft::inverse(fftPrecision* data){

    distribution.getZPencils(data,scratch,d_Buff1,d_Buff2);
    inverse_1d_fft(d_Buff1, Ng, nlocal);
    distribution.returnZPencils(data,scratch,d_Buff1,d_Buff2);

    distribution.getXPencils(data,scratch,d_Buff1,d_Buff2);
    inverse_1d_fft(d_Buff1, Ng, nlocal);
    distribution.returnXPencils(data,scratch,d_Buff1,d_Buff2);

    distribution.getYPencils(data,scratch,d_Buff1,d_Buff2);
    inverse_1d_fft(d_Buff1, Ng, nlocal);
    launch_scale_fft(d_Buff1, Ng, nlocal, blockSize);
    distribution.returnYPencils(data,scratch,d_Buff1,d_Buff2);

}