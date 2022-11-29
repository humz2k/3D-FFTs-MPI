#include "kernels.h"

__global__
void d_z_a2a_to_z_pencils(fftPrecision* source, fftPrecision* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){
        int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_per_rank);
        int mini_pencil_offset = (rank % n_mini_pencils_stacked) * n_cells_mini_pencils;
        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int offset = stack_id * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];
    }
    
}

extern void launch_d_z_a2a_to_z_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    d_z_a2a_to_z_pencils<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, nlocal);

    cudaDeviceSynchronize();
}

__global__
void d_z_pencils_to_z_a2a(fftPrecision* source, fftPrecision* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){
        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_stacked);
        int rank = stack_id * n_mini_pencils_stacked + mini_pencil_id;
        int pencil_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % (n_mini_pencils_per_rank);
        int pencil_offset = pencil_id * n_cells_mini_pencils;
        //int mini_pencil_offset = (mini_pencil_id % n_mini_pencils_stacked) * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];
    }

}

extern void launch_d_z_pencils_to_z_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){
        
    int numBlocks = (nlocal + blockSize - 1) / blockSize;
    
    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    d_z_pencils_to_z_a2a<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, nlocal);

    cudaDeviceSynchronize();

}

__global__ 
void d_x_a2a_to_x_pencils(fftPrecision* source, fftPrecision* dest, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int n_cells_per_stack, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        //int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_cells_per_stack / n_cells_mini_pencils);
        int stack_id = idx / n_cells_per_stack;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils;
        int offset = stack_id * n_cells_mini_pencils;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];

    }

}

extern void launch_d_x_a2a_to_x_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = nlocal / n_mini_pencils_stacked;

    d_x_a2a_to_x_pencils<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, n_cells_per_stack, nlocal);

    cudaDeviceSynchronize();

}

__global__
void d_x_pencils_to_x_a2a(fftPrecision* source, fftPrecision* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int n_cells_per_stack, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int mini_pencil_id = (idx / n_cells_mini_pencils);
        int stack_id = mini_pencil_id % n_mini_pencils_stacked;
        int rank = (idx / (n_cells_per_rank * n_mini_pencils_stacked)) + stack_id * (n_cells_per_stack / n_cells_per_rank);
        int pencil_id = (idx / (n_mini_pencils_stacked * n_cells_mini_pencils)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = pencil_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];

    }

}

extern void launch_d_x_pencils_to_x_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = nlocal / n_mini_pencils_stacked;

    d_x_pencils_to_x_a2a<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, n_cells_per_stack, nlocal);

    cudaDeviceSynchronize();

}

__global__
void d_y_a2a_to_y_pencils(fftPrecision* source, fftPrecision* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int n_ranks_per_tower, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        //int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % ((n_ranks_per_tower * n_mini_pencils_per_rank));
        int stack_id = idx / (n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank);
        int local_stack_id = (idx / (n_ranks_per_tower * n_cells_per_rank)) % n_mini_pencils_stacked;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + local_stack_id * n_cells_mini_pencils;
        int offset = stack_id * n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = mini_pencil_offset + offset + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];

    }

}

extern void launch_d_y_a2a_to_y_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    //int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    d_y_a2a_to_y_pencils<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, n_ranks_per_tower, nlocal);

    cudaDeviceSynchronize();

}

__global__
void d_y_pencils_to_y_a2a(fftPrecision* source, fftPrecision* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked, int n_ranks_per_tower, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int mini_pencil_id = ((idx / n_cells_mini_pencils) % n_mini_pencils_stacked) * n_ranks_per_tower;
        int stack_id = (idx / (n_cells_per_rank)) / n_mini_pencils_stacked;
        int rank = mini_pencil_id + (stack_id) + (stack_id / n_ranks_per_tower) * n_ranks_per_tower;
        int local_stack_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = local_stack_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx*2] = source[idx*2];
        dest[new_idx*2 + 1] = source[idx*2 + 1];

    }

}

extern void launch_d_y_pencils_to_y_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims){

    int numBlocks = (nlocal + blockSize - 1) / blockSize;

    int n_cells_per_rank = nlocal / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    //int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    d_y_pencils_to_y_a2a<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked, n_ranks_per_tower, nlocal);

    cudaDeviceSynchronize();

}