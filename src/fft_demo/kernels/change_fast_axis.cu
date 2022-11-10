__global__
void d_fast_z_to_x(float* source, float* dest, int lgridx, int lgridy, int lgridz){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int i = idx / (lgridx * lgridy);
    int j = (idx - (i * lgridx * lgridy)) / lgridx;
    int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

    int dest_index = i*lgridy*lgridx + j*lgridx + k;
    int source_index = k*lgridy*lgridz + j*lgridz + i;

    dest[dest_index * 2] = source[source_index * 2];
    dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

}

extern "C"{

    void launch_d_fast_z_to_x(float** source, float** dest, int* local_grid_size, int blockSize, int nlocal){

        int numBlocks = (nlocal + blockSize - 1) / blockSize;

        d_fast_z_to_x<<<numBlocks,blockSize>>>(source[0], dest[0], local_grid_size[0], local_grid_size[1], local_grid_size[2]);

        cudaDeviceSynchronize();

    }

}