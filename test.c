#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

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
    grid_size[2] = Ng / dims[1];

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

int main(int argc, char** argv) {

    MPI_Init(NULL, NULL);

    int ndims = 3;
    int Ng = 8;

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int dims[3] = {0,0,0};
    MPI_Dims_create(world_size,ndims,dims);

    int coords[3] = {0,0,0};
    id2coords(world_rank,dims,coords);

    int local_grid_size[3] = {0,0,0};
    grid2top(Ng,dims,local_grid_size);

    int local_coordinates_start[3] = {0,0,0};
    get_local_start(local_grid_size,coords,local_coordinates_start);

    int nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    int* myGridCells = (int*) malloc(nlocal * sizeof(int));
    init_grid_cells(world_rank,local_coordinates_start,local_grid_size,myGridCells,Ng);



    char name[] = "proc0";
    char c = world_rank + '0';
    name[4] = c;

    FILE *out_file = fopen(name, "w");

    fprintf(out_file,"Processor %d, coords [%d,%d,%d]\n",
         world_rank, coords[0], coords[1], coords[2]); 

    int xyz[3] = {0,0,0};

    fprintf(out_file,"step0\n");

    for (int i = 0; i < nlocal; i++){

        int id = myGridCells[i];

        gridID2xyz(id,Ng,xyz);

        fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

    }

    MPI_Alltoall(myGridCells,8,MPI_INT,myGridCells,8,MPI_INT,MPI_COMM_WORLD);

    fprintf(out_file,"step1\n");

    for (int i = 0; i < nlocal; i++){

        int id = myGridCells[i];

        gridID2xyz(id,Ng,xyz);

        fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

    }

    //printf("%d: [%d,%d,%d], %d\n",world_rank,local_coordinates_start[0],local_coordinates_start[1],local_coordinates_start[2], nlocal);

    free(myGridCells);
    MPI_Finalize();

}