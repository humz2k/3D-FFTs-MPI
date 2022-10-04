#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"

void gridID2xyz(int id, int Ng, int* out){

  out[0] = id / (Ng*Ng);
  out[1] = (id - out[0]*(Ng*Ng)) / Ng;
  out[2] = id - out[0]*(Ng*Ng) - out[1]*(Ng);

}

void id2coords(int id, int* dims, int* out){

  out[0] = id / (dims[1]*dims[2]);
  out[1] = (id - out[0]*(dims[1]*dims[2])) / dims[2];
  out[2] = id - out[0]*(dims[1]*dims[2]) - out[1]*(dims[2]);

}

void grid2top(int Ng, int* dims, int* grid_size){

    grid_size[0] = Ng / dims[0];
    grid_size[1] = Ng / dims[1];
    grid_size[2] = Ng / dims[2];

}

void get_local_start(int* local_grid_size, int* coords, int* local_coordinates_start){

    local_coordinates_start[0] = local_grid_size[0] * coords[0];
    local_coordinates_start[1] = local_grid_size[1] * coords[1];
    local_coordinates_start[2] = local_grid_size[2] * coords[2];

}

void init_grid_cells(int world_rank, int* local_coordinates_start, int* local_grid_size, int* myGridCells, int Ng){

    int count = 0;

    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[2]; k++){

                int id = (i + local_coordinates_start[0]) * Ng * Ng + (j + local_coordinates_start[1])*Ng + k + local_coordinates_start[2];

                myGridCells[count] = id;

                count++;

            }
        }
    }

}