#include <stdio.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
    MPI_Init(NULL,NULL);
    if (MPIX_Query_cuda_support()) {
        printf("MPI has CUDA-aware support.\n");
    } else {
        printf("MPI does not have CUDA-aware support.\n");
    }
    MPI_Finalize();
}