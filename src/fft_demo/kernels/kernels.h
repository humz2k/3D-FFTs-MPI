
unsigned long long CPUTimer(unsigned long long start);

void initialize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2, int nlocal, float* myGridCellsBuff1);

void finalize_cuda(float** d_myGridCellsBuff1, float** d_myGridCellsBuff2);


void copy_h2d(float** dest, float* source, int nlocal);

void copy_d2h(float* dest, float** source, int nlocal);


void launch_d_fast_z_to_x(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal);

void launch_d_fast_x_to_y(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal);

void launch_d_fast_y_to_z(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal);


void launch_d_z_a2a_to_z_pencils(float** source, float** dest, int blockSize, int world_size, int n_cells, int* local_grid_size, int* dims);

void launch_d_z_pencils_to_z_a2a(float** source, float** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);


void launch_d_x_a2a_to_x_pencils(float** source, float** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);

void launch_d_x_pencils_to_x_a2a(float** source, float** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);


void launch_d_y_a2a_to_y_pencils(float** source, float** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);
void launch_d_y_pencils_to_y_a2a(float** source, float** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);