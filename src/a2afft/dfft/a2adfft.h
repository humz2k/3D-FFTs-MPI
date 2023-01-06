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

        fftPrecision** d_scratch;

        int plans_made;

        a2adfft(a2aDistribution &dist);

        void make_plans(fftPrecision** input_d_scratch);

        void forward(fftPrecision** d_data);

        void inverse(fftPrecision** d_data);

};