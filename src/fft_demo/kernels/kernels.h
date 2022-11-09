
unsigned long long CPUTimer(unsigned long long start);

void initialize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2, int nlocal, float* myGridCellsBuff1);

void finalize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2);

void copy_h2d(float** dest, float* source, int nlocal);

void copy_d2h(float* dest, float** source, int nlocal);

void launch_d_z_a2a_to_z_pencils(float** source, float** dest, int blockSize, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims);