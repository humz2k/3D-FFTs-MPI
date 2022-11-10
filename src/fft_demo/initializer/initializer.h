extern void rank2coords(int id, int* dims, int* out);

extern void topology2localgrid(int Ng, int* dims, int* grid_size);

extern void get_local_grid_start(int* local_grid_size, int* coords, int* local_coordinates_start);

extern void populate_grid(float* myGridCells, int nlocal);

extern void populate_grid_index(float* myGridCells, int nlocal, int world_rank, int* local_coordinates_start, int* local_grid_size, int Ng);