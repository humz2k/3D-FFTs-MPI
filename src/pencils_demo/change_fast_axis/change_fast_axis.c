#include <stdio.h>
#include <stdlib.h>
#include "change_fast_axis.h"

void fast_z_to_x(int* source, int* dest, int* local_grid_size){
    for (int i = 0; i < local_grid_size[2]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[0]; k++){
                dest[i*local_grid_size[1]*local_grid_size[0] + j*local_grid_size[0] + k] = source[k*local_grid_size[1]*local_grid_size[2] + j*local_grid_size[2] + i];
            }
        }
    }
}

void fast_x_to_y(int* source, int* dest, int* local_grid_size){
    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[2]; j++){
            for (int k = 0; k < local_grid_size[1]; k++){
                dest[i*local_grid_size[2]*local_grid_size[1] + j*local_grid_size[1] + k] = source[j*local_grid_size[0]*local_grid_size[1] + k*local_grid_size[0] + i];
            }
        }
    }
}

void fast_y_to_z(int* source, int* dest, int* local_grid_size){
    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[2]; k++){
                dest[i*local_grid_size[1]*local_grid_size[2] + j*local_grid_size[2] + k] = source[i*local_grid_size[1]*local_grid_size[2] + k*local_grid_size[1] + j];
            }
        }
    }
}