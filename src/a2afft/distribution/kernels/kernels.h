#define fftPrecision double

extern void copy_h2d(fftPrecision** dest, fftPrecision* source, int nlocal);
extern void copy_d2h(fftPrecision* dest, fftPrecision** source, int nlocal);

void launch_d_fast_z_to_x(fftPrecision** source, fftPrecision** dest, int* local_grid_size, int blockSize, int nlocal);
void launch_d_fast_x_to_y(fftPrecision** source, fftPrecision** dest, int* local_grid_size, int blockSize, int nlocal);
void launch_d_fast_y_to_z(fftPrecision** source, fftPrecision** dest, int* local_grid_size, int blockSize, int nlocal);

void launch_d_z_a2a_to_z_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int n_cells, int* local_grid_size, int* dims);
void launch_d_z_pencils_to_z_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);

void launch_d_x_a2a_to_x_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);
void launch_d_x_pencils_to_x_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);

void launch_d_y_a2a_to_y_pencils(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);
void launch_d_y_pencils_to_y_a2a(fftPrecision** source, fftPrecision** dest, int blockSize, int world_size, int nlocal, int* local_grid_size, int* dims);
