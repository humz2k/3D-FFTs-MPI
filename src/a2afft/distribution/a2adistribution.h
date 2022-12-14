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
        int coords[3];
        int local_coordinates_start[3];
        int blockSize;
        MPI_Datatype TYPE_COMPLEX;
        MPI_Comm comm;

        #ifndef cudampi
        fftPrecision* h_scratch1;
        fftPrecision* h_scratch2;
        #endif

        a2aDistribution(MPI_Comm input_comm, int input_Ng, int input_blockSize);

        void getZPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnZPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void getXPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnXPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void getYPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void returnYPencils(fftPrecision** d_Buff1, fftPrecision** d_Buff2);

        void finalize();

};