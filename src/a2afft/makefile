roomba:

	nvcc -c distribution/kernels/a2acudacopy.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2acudacopy.o
	nvcc -c distribution/kernels/a2achangefastaxis.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2achangefastaxis.o
	nvcc -c distribution/kernels/a2apencils.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2apencils.o
	mpicxx -c distribution/a2adistribution.cpp -o a2adistribution.o

	nvcc -c dfft/kernels/a2affts.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2affts.o
	mpicxx -c dfft/a2adfft.cpp -o a2adfft.o

	nvcc -c tests/cuda_utils.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o cuda_utils.o
	mpicxx -c tests/test.cpp -o temp_test.o

	mpicxx temp_test.o cuda_utils.o a2adfft.o a2affts.o a2adistribution.o a2apencils.o a2achangefastaxis.o a2acudacopy.o -L/usr/local/cuda/lib64 -lcufft -lcudart -o test.o


NPROC := 8
NG := 8
BLOCKSIZE := 64

run:
	mpirun -n $(NPROC) ./test.o -Ng $(NG) -blockSize $(BLOCKSIZE)

clean:
	rm *.o

clear_test:
	rm dims
	rm proc*

check_gpu:
	mpicxx tests/gpu_awareness_check.cpp -o gpu_check.o
	mpirun ./gpu_check.o