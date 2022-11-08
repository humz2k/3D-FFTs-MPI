extern void gridID2xyz(int id, int Ng, int* out);

extern void id2coords(int id, int* dims, int* out);

extern void grid2top(int Ng, int* dims, int* grid_size);

extern void get_local_start(int* local_grid_size, int* coords, int* local_coordinates_start);

extern void init_grid_cells(int world_rank, int* local_coordinates_start, int* local_grid_size, int* myGridCells, int Ng);

extern void copy(int* source, int* dest, int len);
