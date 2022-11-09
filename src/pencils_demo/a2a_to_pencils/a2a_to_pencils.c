#include <stdio.h>
#include <stdlib.h>
#include "a2a_to_pencils.h"

//To go from alltoall to pencils
void z_a2a_to_z_pencils(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    for (int idx = 0; idx < n_cells; idx++){

        int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_per_rank);
        int mini_pencil_offset = (rank % n_mini_pencils_stacked) * n_cells_mini_pencils;
        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int offset = stack_id * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + (idx % n_cells_mini_pencils);
        
        dest[new_idx] = source[idx];

    }
}

void z_pencils_to_z_a2a(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    for (int idx = 0; idx < n_cells; idx++){

        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_stacked);
        int rank = stack_id * n_mini_pencils_stacked + mini_pencil_id;
        int pencil_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % (n_mini_pencils_per_rank);
        int pencil_offset = pencil_id * n_cells_mini_pencils;
        int mini_pencil_offset = (mini_pencil_id % n_mini_pencils_stacked) * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];
    }
}

void x_a2a_to_x_pencils(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;

    for (int idx = 0; idx < n_cells; idx++){

        int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_cells_per_stack / n_cells_mini_pencils);
        int stack_id = idx / n_cells_per_stack;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils;
        int offset = stack_id * n_cells_mini_pencils;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

    }
}

void x_pencils_to_x_a2a(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;

    for (int idx = 0; idx < n_cells; idx++){

        int mini_pencil_id = (idx / n_cells_mini_pencils);
        int stack_id = mini_pencil_id % n_mini_pencils_stacked;
        int rank = (idx / (n_cells_per_rank * n_mini_pencils_stacked)) + stack_id * (n_cells_per_stack / n_cells_per_rank);
        int pencil_id = (idx / (n_mini_pencils_stacked * n_cells_mini_pencils)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = pencil_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

    }
}

void y_a2a_to_y_pencils(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    for (int idx = 0; idx < n_cells; idx++){

        int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % ((n_ranks_per_tower * n_mini_pencils_per_rank));
        int stack_id = idx / (n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank);
        int local_stack_id = (idx / (n_ranks_per_tower * n_cells_per_rank)) % n_mini_pencils_stacked;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + local_stack_id * n_cells_mini_pencils;
        int offset = stack_id * n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = mini_pencil_offset + offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

    }
}

void y_pencils_to_y_a2a(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    for (int idx = 0; idx < n_cells; idx++){
        

        int mini_pencil_id = ((idx / n_cells_mini_pencils) % n_mini_pencils_stacked) * n_ranks_per_tower;
        int stack_id = (idx / (n_cells_per_rank)) / n_mini_pencils_stacked;
        int rank = mini_pencil_id + (stack_id) + (stack_id / n_ranks_per_tower) * n_ranks_per_tower;
        int local_stack_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = local_stack_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

    }
}