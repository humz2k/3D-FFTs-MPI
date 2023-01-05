#define fftPrecision double

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "kernels/kernels.h"
#include "../distribution/a2adistribution.h"

class a2adfft {

    public:
        int Ng;
        int nlocal;
        int world_size;
        int blockSize;
        a2aDistribution &distribution;

        fftPrecision* scratch;
        fftPrecision** d_Buff1;
        fftPrecision** d_Buff2;

        int plans_made;

        a2adfft(a2aDistribution &dist);

        void make_plans(fftPrecision* input_scratch, fftPrecision** input_d_Buff1, fftPrecision** input_d_Buff2);

        void forward(fftPrecision* data);

        void inverse(fftPrecision* data);

};