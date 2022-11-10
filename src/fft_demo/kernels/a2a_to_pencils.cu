__global__
void d_z_a2a_to_z_pencils(float* source, float* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int rank = idx / n_cells_per_rank;
    int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_per_rank);
    int mini_pencil_offset = (rank % n_mini_pencils_stacked) * n_cells_mini_pencils;
    int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
    int offset = stack_id * n_mini_pencils_stacked * n_cells_per_rank;
    int new_idx = offset + mini_pencil_offset + mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + (idx % n_cells_mini_pencils);

    dest[new_idx*2] = source[idx*2];
    dest[new_idx*2 + 1] = source[idx*2 + 1];
    
}

extern "C" {
    void launch_d_z_a2a_to_z_pencils(float** source, float** dest, int blockSize, int world_size, int world_rank, int nlocal, int* local_grid_size, int* dims){

        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        int n_cells_per_rank = nlocal / world_size;
        int n_cells_mini_pencils = local_grid_size[2];
        int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
        int n_mini_pencils_stacked = dims[2];

        d_z_a2a_to_z_pencils<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked);

        cudaDeviceSynchronize();
    }
}

__global__
void d_z_pencils_to_z_a2a(float* source, float* dest, int n_cells_per_rank, int n_cells_mini_pencils, int n_mini_pencils_per_rank, int n_mini_pencils_stacked){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
    int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_stacked);
    int rank = stack_id * n_mini_pencils_stacked + mini_pencil_id;
    int pencil_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % (n_mini_pencils_per_rank);
    int pencil_offset = pencil_id * n_cells_mini_pencils;
    int mini_pencil_offset = (mini_pencil_id % n_mini_pencils_stacked) * n_cells_mini_pencils;
    int offset = rank * n_cells_per_rank;
    int new_idx = offset + pencil_offset + (idx % n_cells_mini_pencils);

    dest[new_idx*2] = source[idx*2];
    dest[new_idx*2 + 1] = source[idx*2 + 1];

}

extern "C" {
    void launch_d_z_pencils_to_z_a2a(float** source, float** dest, int blockSize, int world_size, int world_rank, int nlocal, int* local_grid_size, int* dims){
        int n_cells_per_rank = nlocal / world_size;
        int n_cells_mini_pencils = local_grid_size[2];
        int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
        int n_mini_pencils_stacked = dims[2];

        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        d_z_pencils_to_z_a2a<<<numBlocks,blockSize>>>(source[0], dest[0], n_cells_per_rank, n_cells_mini_pencils, n_mini_pencils_per_rank, n_mini_pencils_stacked);

        cudaDeviceSynchronize();

    }
}