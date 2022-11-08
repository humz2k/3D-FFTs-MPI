#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers/helpers.h"
#include "a2a_to_pencils/a2a_to_pencils.h"
#include "change_fast_axis/change_fast_axis.h"

void save(FILE* out_file, int step, int* xyz, int Ng, int nlocal, int* myGridCells){
    fprintf(out_file,"step%d\n",step);
    for (int i = 0; i < nlocal; i++){
        int id = myGridCells[i];
        gridID2xyz(id,Ng,xyz);
        fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);
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
    //dims[0] = 1; dims[1] = 2; dims[2] = 3;

    int coords[3] = {0,0,0}; id2coords(world_rank,dims,coords);
    int local_grid_size[3] = {0,0,0}; grid2top(Ng,dims,local_grid_size);

    int local_coordinates_start[3] = {0,0,0}; get_local_start(local_grid_size,coords,local_coordinates_start);
    int nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    if (world_rank == 0){
        printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);
        printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("NLOCAL: %d\n",nlocal);
    }

    //initialize grid
    int* myGridCellsBuff1 = (int*) malloc(nlocal * sizeof(int)); 
    int* myGridCellsBuff2 = (int*) malloc(nlocal * sizeof(int)); 
    init_grid_cells(world_rank,local_coordinates_start,local_grid_size,myGridCellsBuff1,Ng);

    //create file
    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");
    fprintf(out_file,"Processor %d, coords [%d,%d,%d]\n",
         world_rank, coords[0], coords[1], coords[2]); 

    int xyz[3] = {0,0,0};
    int nsends;

    save(out_file,0,xyz,Ng,nlocal,myGridCellsBuff1);

    //
    //Make pencils along Z axis.
    //
    nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,1,xyz,Ng,nlocal,myGridCellsBuff2);

    z_a2a_to_z_pencils(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,2,xyz,Ng,nlocal,myGridCellsBuff1);

    z_pencils_to_z_a2a(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,3,xyz,Ng,nlocal,myGridCellsBuff2);

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,4,xyz,Ng,nlocal,myGridCellsBuff1);

    //
    //Make Pencils along x axis
    //
    fast_z_to_x(myGridCellsBuff1,myGridCellsBuff2,local_grid_size);
    save(out_file,5,xyz,Ng,nlocal,myGridCellsBuff2);

    nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,6,xyz,Ng,nlocal,myGridCellsBuff1);

    x_a2a_to_x_pencils(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,7,xyz,Ng,nlocal,myGridCellsBuff2);

    x_pencils_to_x_a2a(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,8,xyz,Ng,nlocal,myGridCellsBuff1);

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,9,xyz,Ng,nlocal,myGridCellsBuff2);

    //
    // Make pencils along y axis
    //
    fast_x_to_y(myGridCellsBuff2,myGridCellsBuff1,local_grid_size);
    save(out_file,10,xyz,Ng,nlocal,myGridCellsBuff1);

    nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,11,xyz,Ng,nlocal,myGridCellsBuff2);

    y_a2a_to_y_pencils(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,12,xyz,Ng,nlocal,myGridCellsBuff1);

    y_pencils_to_y_a2a(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,13,xyz,Ng,nlocal,myGridCellsBuff2);

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,14,xyz,Ng,nlocal,myGridCellsBuff1);

    fast_y_to_z(myGridCellsBuff1,myGridCellsBuff2,local_grid_size);
    save(out_file,15,xyz,Ng,nlocal,myGridCellsBuff2);

    free(myGridCellsBuff1);
    free(myGridCellsBuff2);
    MPI_Finalize();

}