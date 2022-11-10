__global__
void d_fast_z_to_x(float* source, float* dest, int lgridx, int lgridy, int lgridz, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i*lgridy*lgridx + j*lgridx + k;
        int source_index = k*lgridy*lgridz + j*lgridz + i;

        dest[dest_index * 2] = source[source_index * 2];
        dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

extern "C"{

    void launch_d_fast_z_to_x(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal){

        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        d_fast_z_to_x<<<numBlocks,blockSize>>>(source[0], dest[0], local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

        cudaDeviceSynchronize();

    }

}

__global__
void d_fast_x_to_y(float* source, float* dest, int lgridx, int lgridy, int lgridz, int nlocal){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridy;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridy);

        int dest_index = i*lgridz*lgridy + j*lgridy + k;
        int source_index = j*lgridx*lgridy + k*lgridx + i;

        dest[dest_index * 2] = source[source_index * 2];
        dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

extern "C" {

    void launch_d_fast_x_to_y(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal){
        
        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        d_fast_x_to_y<<<numBlocks,blockSize>>>(source[0], dest[0], local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

        cudaDeviceSynchronize();

    }

}

__global__
void d_fast_y_to_z(float* source, float* dest, int lgridx, int lgridy, int lgridz, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridz;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridz);

        int dest_index = i*lgridz*lgridy + j*lgridz + k;
        int source_index = i*lgridz*lgridy + k*lgridy + j;

        dest[dest_index * 2] = source[source_index * 2];
        dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

extern "C" {

    void launch_d_fast_y_to_z(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal){
        
        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        d_fast_y_to_z<<<numBlocks,blockSize>>>(source[0], dest[0], local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

        cudaDeviceSynchronize();

    }

}