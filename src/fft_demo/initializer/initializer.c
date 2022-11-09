#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "initializer.h"

extern void rank2coords(int rank, int* dims, int* out){

  out[0] = rank / (dims[1]*dims[2]);
  out[1] = (rank - out[0]*(dims[1]*dims[2])) / dims[2];
  out[2] = rank - out[0]*(dims[1]*dims[2]) - out[1]*(dims[2]);

}

extern void topology2localgrid(int Ng, int* dims, int* grid_size){

    grid_size[0] = Ng / dims[0];
    grid_size[1] = Ng / dims[1];
    grid_size[2] = Ng / dims[2];

}

extern void get_local_grid_start(int* local_grid_size, int* coords, int* local_coordinates_start){

    local_coordinates_start[0] = local_grid_size[0] * coords[0];
    local_coordinates_start[1] = local_grid_size[1] * coords[1];
    local_coordinates_start[2] = local_grid_size[2] * coords[2];

}

extern void populate_grid(float* myGridCells, int nlocal, int world_rank, int* local_coordinates_start, int* local_grid_size, int Ng){

    /*for (int i = 0; i < (nlocal*2); i+=2){
        myGridCells[i] = ((float)rand()) / (float)RAND_MAX;
        myGridCells[i+1] = 0;
    }*/

    int count = 0;

    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[2]; k++){

                int id = (i + local_coordinates_start[0]) * Ng * Ng + (j + local_coordinates_start[1])*Ng + k + local_coordinates_start[2];

                myGridCells[count*2] = (float)id;
                myGridCells[count*2 + 1] = 0;

                count++;

            }
        }
    }

}