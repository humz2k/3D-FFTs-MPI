#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers/helpers.h"

void save(FILE* out_file, int step, int* xyz, int Ng, int nlocal, int* myGridCells){

    fprintf(out_file,"step%d\n",step);

    for (int i = 0; i < nlocal; i++){

        int id = myGridCells[i];

        gridID2xyz(id,Ng,xyz);

        fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

    }

}

void copy(int* source, int* dest, int len){

    for (int i = 0; i < len; i++){
        dest[i] = source[i];
    }

}

void fix_1(int* source, int* dest, int* local_grid_size, int n_cells, int* dims){

    for (int idx = 0; idx < n_cells; idx++){

        int n_cell_in_cycles = (n_cells / dims[0]);
        int n_cell_in_lines = n_cells / (local_grid_size[0] * local_grid_size[1]);

        int n_lines_in_cycle = n_cell_in_cycles / n_cell_in_lines;

        int cycleid = idx / n_cell_in_cycles;
        int lineid = (idx / n_cell_in_lines);

        int newlineid = (lineid % 2) + 2 * (lineid / 4);

        dest[newlineid * n_cell_in_lines] = source[idx];

    }

}

void fix_2_forward(int* source, int* dest, int* local_grid_size, int n_cells, int* dims){

}

void fix_2_backward(int* source, int* dest, int* local_grid_size, int n_cells){

}

void transpose_1(int* source, int* dest, int* local_grid_size){

    for (int i = 0; i < local_grid_size[2]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[0]; k++){
                dest[i*local_grid_size[1]*local_grid_size[0] + j*local_grid_size[0] + k] = source[k*local_grid_size[1]*local_grid_size[2] + j*local_grid_size[2] + i];
            }
        }
    }
}

void transpose_2(int* source, int* dest, int* local_grid_size){

    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[2]; j++){
            for (int k = 0; k < local_grid_size[1]; k++){
                dest[i*local_grid_size[2]*local_grid_size[1] + j*local_grid_size[1] + k] = source[j*local_grid_size[0]*local_grid_size[1] + k*local_grid_size[0] + i];
            }
        }
    }
}

void transpose_3(int* source, int* dest, int* local_grid_size){

    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[2]; k++){
                dest[i*local_grid_size[1]*local_grid_size[2] + j*local_grid_size[2] + k] = source[i*local_grid_size[1]*local_grid_size[2] + k*local_grid_size[1] + j];
            }
        }
    }
}

int main(int argc, char** argv) {

    MPI_Init(NULL, NULL);

    int ndims = 3;
    int Ng = 8;
    
    //get MPI world rank and size
    int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    //get info about dimensions/local grid size
    int dims[3] = {0,0,0}; MPI_Dims_create(world_size,ndims,dims);
    /*dims[0] = 1;
    dims[1] = 2;
    dims[2] = 3;*/

    printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);

    int coords[3] = {0,0,0}; id2coords(world_rank,dims,coords);
    int local_grid_size[3] = {0,0,0}; grid2top(Ng,dims,local_grid_size);

    printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    
    int local_coordinates_start[3] = {0,0,0}; get_local_start(local_grid_size,coords,local_coordinates_start);
    int nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    printf("NLOCAL: %d\n",nlocal);

    //initialize grid
    int* myGridCells = (int*) malloc(nlocal * sizeof(int)); 
    int* myGridCellsTemp = (int*) malloc(nlocal * sizeof(int)); 
    init_grid_cells(world_rank,local_coordinates_start,local_grid_size,myGridCells,Ng);

    //create file
    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");
    fprintf(out_file,"Processor %d, coords [%d,%d,%d]\n",
         world_rank, coords[0], coords[1], coords[2]); 
    int xyz[3] = {0,0,0};

    save(out_file,0,xyz,Ng,nlocal,myGridCells);

    int nsends;

    nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(myGridCells,nsends,MPI_INT,myGridCellsTemp,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,1,xyz,Ng,nlocal,myGridCellsTemp);

    fix_1(myGridCellsTemp,myGridCells,local_grid_size,nlocal,dims);
    save(out_file,2,xyz,Ng,nlocal,myGridCells);

    fix_1(myGridCells,myGridCellsTemp,local_grid_size,nlocal,dims);
    save(out_file,3,xyz,Ng,nlocal,myGridCellsTemp);

    MPI_Alltoall(myGridCellsTemp,nsends,MPI_INT,myGridCells,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,4,xyz,Ng,nlocal,myGridCells);

    transpose_1(myGridCells,myGridCellsTemp,local_grid_size);
    save(out_file,5,xyz,Ng,nlocal,myGridCellsTemp);

    nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];
    MPI_Alltoall(myGridCellsTemp,nsends,MPI_INT,myGridCells,nsends,MPI_INT,MPI_COMM_WORLD);
    
    //copy(myGridCells,myGridCellsTemp,nlocal);
    save(out_file,6,xyz,Ng,nlocal,myGridCells);

    fix_2_forward(myGridCells,myGridCellsTemp,local_grid_size,nlocal,dims);
    save(out_file,7,xyz,Ng,nlocal,myGridCellsTemp);

    fix_2_backward(myGridCellsTemp,myGridCells,local_grid_size,nlocal);
    save(out_file,8,xyz,Ng,nlocal,myGridCells);

    MPI_Alltoall(myGridCellsTemp,nsends,MPI_INT,myGridCells,nsends,MPI_INT,MPI_COMM_WORLD);
    transpose_2(myGridCells,myGridCellsTemp,local_grid_size);
    save(out_file,9,xyz,Ng,nlocal,myGridCellsTemp);

    nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];
    MPI_Alltoall(myGridCellsTemp,nsends,MPI_INT,myGridCells,nsends,MPI_INT,MPI_COMM_WORLD);
    copy(myGridCells,myGridCellsTemp,nlocal);
    save(out_file,10,xyz,Ng,nlocal,myGridCellsTemp);

    MPI_Alltoall(myGridCellsTemp,nsends,MPI_INT,myGridCells,nsends,MPI_INT,MPI_COMM_WORLD);
    transpose_3(myGridCells,myGridCellsTemp,local_grid_size);
    save(out_file,11,xyz,Ng,nlocal,myGridCellsTemp);

    free(myGridCells);
    free(myGridCellsTemp);
    MPI_Finalize();

}