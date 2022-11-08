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

//To go from alltoall to pencils
void fix_1_forward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    //if (world_rank == 0){
    //    printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked);
    //}

    for (int idx = 0; idx < n_cells; idx++){
        int rank = idx / n_cells_per_rank;
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_per_rank);
        int mini_pencil_offset = (rank % n_mini_pencils_stacked) * n_cells_mini_pencils;
        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int offset = stack_id * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + (idx % n_cells_mini_pencils);
        
        dest[new_idx] = source[idx];

        //if (world_rank == 0){
        //    printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, mini_pencil_offset: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,mini_pencil_offset,new_idx);
        //}
    }
}

void fix_1_backward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[2];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[2];

    /*if (world_rank == 0){
        printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked);
    }*/

    for (int idx = 0; idx < n_cells; idx++){

        int stack_id = idx / (n_mini_pencils_stacked * n_cells_per_rank);
        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_mini_pencils_stacked);
        int rank = stack_id * n_mini_pencils_stacked + mini_pencil_id;

        int pencil_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % (n_mini_pencils_per_rank);
        int pencil_offset = pencil_id * n_cells_mini_pencils;

        int mini_pencil_offset = (mini_pencil_id % n_mini_pencils_stacked) * n_cells_mini_pencils;

        int offset = rank * n_cells_per_rank;

        int new_idx = offset + pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

        /*if (world_rank == 0){
            printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, pencil_id: %d, pencil_offset: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,pencil_id, pencil_offset,new_idx);
        }*/
    }
}

void fix_2_forward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;

    /*if (world_rank == 0){
        printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked);
    }*/

    for (int idx = 0; idx < n_cells; idx++){
        int rank = idx / n_cells_per_rank;

        int mini_pencil_id = (idx / n_cells_mini_pencils) % (n_cells_per_stack / n_cells_mini_pencils);
        int stack_id = idx / n_cells_per_stack;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils;
        int offset = stack_id * n_cells_mini_pencils;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

        /*if (world_rank == 0){
            printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, mini_pencil_offset: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,mini_pencil_offset,new_idx);
        }*/
    }
}

void fix_2_backward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[0];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[0];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;

    /*if (world_rank == 0){
        printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked);
    }*/

    for (int idx = 0; idx < n_cells; idx++){

        int mini_pencil_id = (idx / n_cells_mini_pencils);
        int stack_id = mini_pencil_id % n_mini_pencils_stacked;
        int rank = (idx / (n_cells_per_rank * n_mini_pencils_stacked)) + stack_id * (n_cells_per_stack / n_cells_per_rank);

        int pencil_id = (idx / (n_mini_pencils_stacked * n_cells_mini_pencils)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = pencil_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

        /*if (world_rank == 0){
            printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, mini_pencil_offset: %d, pencil_id: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,mini_pencil_offset, pencil_id, new_idx);
        }*/
    }
}

void fix_3_forward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    /*if (world_rank == 0){
        printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\nn_ranks_per_stack: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked,n_ranks_per_stack);
    }*/

    for (int idx = 0; idx < n_cells; idx++){
        int rank = idx / n_cells_per_rank;

        int mini_pencil_id = (idx / n_cells_mini_pencils) % ((n_ranks_per_tower * n_mini_pencils_per_rank));
        int stack_id = idx / (n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank);
        int local_stack_id = (idx / (n_ranks_per_tower * n_cells_per_rank)) % n_mini_pencils_stacked;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + local_stack_id * n_cells_mini_pencils;
        int offset = stack_id * n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = mini_pencil_offset + offset + (idx % n_cells_mini_pencils);

        dest[new_idx] = source[idx];

        /*if (world_rank == 0){
            printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, mini_pencil_offset: %d, local_stack_id: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,mini_pencil_offset,local_stack_id,new_idx);
        }*/
    }
}

void fix_3_backward(int* source, int* dest, int world_size, int world_rank, int n_cells, int* local_grid_size, int* dims){
    int n_cells_per_rank = n_cells / world_size;
    int n_cells_mini_pencils = local_grid_size[1];
    int n_mini_pencils_per_rank = n_cells_per_rank / n_cells_mini_pencils;
    int n_mini_pencils_stacked = dims[1];
    int n_cells_per_stack = n_cells / n_mini_pencils_stacked;
    int n_ranks_per_stack = world_size / n_mini_pencils_stacked;
    int n_ranks_per_tower = (n_ranks_per_stack / dims[0]);

    if (world_rank == 0){
        printf("n_cells_per_rank: %d\nn_cells_mini_pencils: %d\nn_mini_pencils_per_rank: %d\nn_mini_pencils_stacked: %d\nn_ranks_per_stack: %d\n",n_cells_per_rank,n_cells_mini_pencils,n_mini_pencils_per_rank,n_mini_pencils_stacked,n_ranks_per_stack);
    }

    for (int idx = 0; idx < n_cells; idx++){
        

        int mini_pencil_id = ((idx / n_cells_mini_pencils) % n_mini_pencils_stacked) * n_ranks_per_tower;
        int stack_id = (idx / (n_cells_per_rank)) / n_mini_pencils_stacked;

        //int rank = mini_pencil_id + stack_id + (idx / (n_cells_per_stack)) * 2;
        int rank = mini_pencil_id + (stack_id) + (stack_id / n_ranks_per_tower) * n_ranks_per_tower;

        int local_stack_id = (idx / (n_cells_mini_pencils * n_mini_pencils_stacked)) % n_mini_pencils_per_rank;
        int mini_pencil_offset = local_stack_id * n_cells_mini_pencils;
        int offset = rank * n_cells_per_rank;
        int new_idx = offset + mini_pencil_offset + (idx % n_cells_mini_pencils);

        /*
        int mini_pencil_id = (idx / n_cells_mini_pencils) % ((n_ranks_per_tower * n_mini_pencils_per_rank));
        int stack_id = idx / (n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank);
        int local_stack_id = (idx / (n_ranks_per_tower * n_cells_per_rank)) % n_mini_pencils_stacked;
        int mini_pencil_offset = mini_pencil_id * n_mini_pencils_stacked * n_cells_mini_pencils + local_stack_id * n_cells_mini_pencils;
        int offset = stack_id * n_ranks_per_tower * n_mini_pencils_stacked * n_cells_per_rank;
        int new_idx = mini_pencil_offset + offset + (idx % n_cells_mini_pencils);
        */

        dest[new_idx] = source[idx];

        if (world_rank == 0){
            printf("idx: %d, rank: %d, offset: %d, mini_pencil_id: %d, stack_id: %d, mini_pencil_offset: %d, local_stack_id: %d, new_idx: %d\n",idx,rank,offset,mini_pencil_id,stack_id,mini_pencil_offset,local_stack_id,new_idx);
        }
    }
}


//change fast index
void transpose_1(int* source, int* dest, int* local_grid_size){
    for (int i = 0; i < local_grid_size[2]; i++){
        for (int j = 0; j < local_grid_size[1]; j++){
            for (int k = 0; k < local_grid_size[0]; k++){
                dest[i*local_grid_size[1]*local_grid_size[0] + j*local_grid_size[0] + k] = source[k*local_grid_size[1]*local_grid_size[2] + j*local_grid_size[2] + i];
            }
        }
    }
}

//change fast index
void transpose_2(int* source, int* dest, int* local_grid_size){
    for (int i = 0; i < local_grid_size[0]; i++){
        for (int j = 0; j < local_grid_size[2]; j++){
            for (int k = 0; k < local_grid_size[1]; k++){
                dest[i*local_grid_size[2]*local_grid_size[1] + j*local_grid_size[1] + k] = source[j*local_grid_size[0]*local_grid_size[1] + k*local_grid_size[0] + i];
            }
        }
    }
}

//change fast index
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
    int* myGridCellsBuff1 = (int*) malloc(nlocal * sizeof(int)); 
    int* myGridCellsBuff2 = (int*) malloc(nlocal * sizeof(int)); 
    init_grid_cells(world_rank,local_coordinates_start,local_grid_size,myGridCellsBuff1,Ng);

    //create file
    char name[] = "proc0"; char c = world_rank + '0'; name[4] = c;
    FILE *out_file = fopen(name, "w");
    fprintf(out_file,"Processor %d, coords [%d,%d,%d]\n",
         world_rank, coords[0], coords[1], coords[2]); 
    int xyz[3] = {0,0,0};

    save(out_file,0,xyz,Ng,nlocal,myGridCellsBuff1);

    int nsends;

    nsends = ((local_grid_size[0] * local_grid_size[1])/world_size) * local_grid_size[2];

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,1,xyz,Ng,nlocal,myGridCellsBuff2);

    fix_1_forward(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,2,xyz,Ng,nlocal,myGridCellsBuff1);

    fix_1_backward(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,3,xyz,Ng,nlocal,myGridCellsBuff2);

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,4,xyz,Ng,nlocal,myGridCellsBuff1);

    
    transpose_1(myGridCellsBuff1,myGridCellsBuff2,local_grid_size);
    save(out_file,5,xyz,Ng,nlocal,myGridCellsBuff2);

    
    nsends = ((local_grid_size[2] * local_grid_size[1])/world_size) * local_grid_size[0];

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,6,xyz,Ng,nlocal,myGridCellsBuff1);

    fix_2_forward(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,7,xyz,Ng,nlocal,myGridCellsBuff2);

    fix_2_backward(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,8,xyz,Ng,nlocal,myGridCellsBuff1);

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,9,xyz,Ng,nlocal,myGridCellsBuff2);

    transpose_2(myGridCellsBuff2,myGridCellsBuff1,local_grid_size);
    save(out_file,10,xyz,Ng,nlocal,myGridCellsBuff1);

    nsends = ((local_grid_size[2] * local_grid_size[0])/world_size) * local_grid_size[1];

    MPI_Alltoall(myGridCellsBuff1,nsends,MPI_INT,myGridCellsBuff2,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,11,xyz,Ng,nlocal,myGridCellsBuff2);

    fix_3_forward(myGridCellsBuff2,myGridCellsBuff1,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,12,xyz,Ng,nlocal,myGridCellsBuff1);

    fix_3_backward(myGridCellsBuff1,myGridCellsBuff2,world_size,world_rank,nlocal,local_grid_size,dims);
    save(out_file,13,xyz,Ng,nlocal,myGridCellsBuff2);

    MPI_Alltoall(myGridCellsBuff2,nsends,MPI_INT,myGridCellsBuff1,nsends,MPI_INT,MPI_COMM_WORLD);
    save(out_file,14,xyz,Ng,nlocal,myGridCellsBuff1);

    transpose_3(myGridCellsBuff1,myGridCellsBuff2,local_grid_size);
    save(out_file,15,xyz,Ng,nlocal,myGridCellsBuff2);

    /*

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
    */

    free(myGridCellsBuff1);
    free(myGridCellsBuff2);
    MPI_Finalize();

}