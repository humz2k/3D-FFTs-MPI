// Author: Wes Kendall
// Copyright 2011 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// An intro MPI hello world program that uses MPI_Init, MPI_Comm_size,
// MPI_Comm_rank, MPI_Finalize, and MPI_Get_processor_name.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void id2xyz(int id, int Ng, int* out){

  out[0] = id / (Ng*Ng);
  out[1] = (id - out[0]*(Ng*Ng)) / Ng;
  out[2] = id - out[0]*(Ng*Ng) - out[1]*(Ng);

}

int main(int argc, char** argv) {
  // Initialize the MPI environment. The two arguments to MPI Init are not
  // currently used by MPI implementations, but are there in case future
  // implementations might need the arguments.
  MPI_Init(NULL, NULL);

  int topology[3];

  topology[0] = 2;
  topology[1] = 2;
  topology[2] = 2;

  int Ng = 8;

  int grid_top[3];
  grid_top[0] = Ng / topology[0];
  grid_top[1] = Ng / topology[1];
  grid_top[2] = Ng / topology[2];

  int myNg = grid_top[0] * grid_top[1] * grid_top[2];

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  int coordinates[3];

  coordinates[0] = world_rank / (topology[1] * topology[2]);
  coordinates[1] = (world_rank - coordinates[0] * (topology[1] * topology[2])) / topology[2];
  coordinates[2] = world_rank - coordinates[0] * (topology[1] * topology[2]) - coordinates[1] * topology[2];

  // Print off a hello world message

  int grid_start[3];
  grid_start[0] = grid_top[0] * coordinates[0];
  grid_start[1] = grid_top[1] * coordinates[1];
  grid_start[2] = grid_top[2] * coordinates[2];

  char name[] = "proc0";
  char c = world_rank + '0';
  name[4] = c;

  FILE *out_file = fopen(name, "w");

  fprintf(out_file,"Processor %d, coords [%d,%d,%d]\n",
         world_rank, coordinates[0], coordinates[1], coordinates[2]);    

  int *xyz = (int*) malloc(3 * sizeof(int));

  int *my_data = (int*) malloc(myNg * sizeof(int));

  int count = 0;

  for (int i = 0; i < grid_top[0]; i++){
    for (int j = 0; j < grid_top[1]; j++){
      for (int k = 0; k < grid_top[2]; k++){

        int id = (i + grid_start[0]) * Ng * Ng + (j + grid_start[1])*Ng + k + grid_start[2];

        my_data[count] = id;

        //id2xyz(id,Ng,xyz);

        //fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

        count++;

      }
    }
  }

  fprintf(out_file,"step0\n");   

  for (int i = 0; i < myNg; i++){

    int id = my_data[i];

    id2xyz(id,Ng,xyz);

    fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

  }

  MPI_Alltoall(my_data,8,MPI_INT,my_data,8,MPI_INT,MPI_COMM_WORLD);

  int n_pencils = (Ng * Ng)/world_size;

  printf("n_pencils %d\n",n_pencils);

  int depth = Ng;
  
  for (int i = 0; i < n_pencils; i+=2){

    int pencil_start = i*depth;
    int swap1 = pencil_start + depth/2;
    int swap2 = pencil_start + depth;
    
    for (int j = 0; j < depth/2; j++){

      int temp = my_data[swap1 + j];
      my_data[swap1 + j] = my_data[swap2 + j];
      my_data[swap2 + j] = temp;

    }

  }

  fprintf(out_file,"step1\n");   

  for (int i = 0; i < myNg; i++){

    int id = my_data[i];

    id2xyz(id,Ng,xyz);

    fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

  }



  int send_counts_one[8] = {4,4,4,4,4,4,4,4};
  int send_disp_one[8] = {0,4,8,12,16,20,24,28};

  int send_counts_two[8] = {4,4,4,4,4,4,4,4};
  int send_disp_two[8] = {0+32,4+32,8+32,12+32,16+32,20+32,24+32,28+32};

  int *my_data2 = (int*) malloc(myNg * sizeof(int));

  MPI_Alltoallv(my_data,send_counts_one,send_disp_one,MPI_INT,my_data2,send_counts_one,send_disp_one,MPI_INT,MPI_COMM_WORLD);

  MPI_Alltoallv(my_data,send_counts_two,send_disp_two,MPI_INT,my_data2,send_counts_two,send_disp_two,MPI_INT,MPI_COMM_WORLD);

  //MPI_Alltoallv((my_data+4),send_counts_two,send_disp_two,MPI_INT,my_data,send_counts_two,send_disp_two,MPI_INT,MPI_COMM_WORLD);

  //MPI_Alltoall(my_data,8,MPI_INT,my_data,8,MPI_INT,MPI_COMM_WORLD);
  //MPI_Alltoall(my_data,8,MPI_INT,my_data,4,MPI_INT,MPI_COMM_WORLD);
  //MPI_Alltoall((my_data + myNg/2),4,MPI_INT,(my_data + myNg/2),4,MPI_INT,MPI_COMM_WORLD);


  fprintf(out_file,"step2\n");   

  for (int i = 0; i < myNg; i++){

    int id = my_data2[i];

    id2xyz(id,Ng,xyz);

    fprintf(out_file,"%d,%d,%d\n",xyz[0],xyz[1],xyz[2]);

  }

  free(xyz);

  fclose(out_file);

  free(my_data);
  free(my_data2);
  // Finalize the MPI environment. No more MPI calls can be made after this
  MPI_Finalize();
}