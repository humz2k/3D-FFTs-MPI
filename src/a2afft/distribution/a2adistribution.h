#define fftPrecision double
#include <mpi.h>


class a2aDistribution {

    public:
        int ndims;
        int Ng;
        int nlocal;
        int world_size;
        int world_rank;
        int local_grid_size[3];
        int dims[3];
        int blockSize;
        MPI_Datatype TYPE_COMPLEX;
        MPI_Comm comm;

        a2aDistribution(MPI_Comm input_comm, int input_Ng);

        void getZPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnZPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void getXPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnXPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void getYPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnYPencils(fftPrecision* data, fftPrecision* scratch, fftPrecision** d_Buff1, fftPrecision** d_Buff2);

};