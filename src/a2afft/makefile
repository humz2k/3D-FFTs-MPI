MY_MPI_CUDA_FLAGS ?= -lcufft -lcudart
MY_CUDA_CC ?= nvcc
MY_MPI_CXX ?= mpicxx
MY_CUDA_FLAGS ?= -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared
MY_MPI_FLAGS ?=
CUDA_LIB_DIR ?= /usr/local/cuda/lib64

VERBOSE ?= FALSE
ifeq ($(VERBOSE),TRUE)
IS_VERBOSE = -Dverbose
else
IS_VERBOSE =
endif

main:

	$(MY_CUDA_CC) -c distribution/kernels/a2acudacopy.cu $(MY_CUDA_FLAGS) -o a2acudacopy.o
	$(MY_CUDA_CC) -c distribution/kernels/a2achangefastaxis.cu $(MY_CUDA_FLAGS) -o a2achangefastaxis.o
	$(MY_CUDA_CC) -c distribution/kernels/a2apencils.cu $(MY_CUDA_FLAGS) -o a2apencils.o
	$(MY_MPI_CXX) -c distribution/a2adistribution.cpp $(MY_MPI_FLAGS) $(IS_VERBOSE) -o a2adistribution.o

	$(MY_CUDA_CC) -c dfft/kernels/a2affts.cu $(MY_CUDA_FLAGS) -o a2affts.o
	$(MY_MPI_CXX) -c dfft/a2adfft.cpp $(MY_MPI_FLAGS) $(IS_VERBOSE) -o a2adfft.o

	$(MY_CUDA_CC) -c tests/cuda_utils.cu $(MY_CUDA_FLAGS) -o cuda_utils.o
	$(MY_MPI_CXX) -c tests/test.cpp $(MY_MPI_FLAGS) -o temp_test.o

	$(MY_MPI_CXX) $(MY_MPI_FLAGS) $(IS_VERBOSE) temp_test.o cuda_utils.o a2adfft.o a2affts.o a2adistribution.o a2apencils.o a2achangefastaxis.o a2acudacopy.o -L$(CUDA_LIB_DIR) $(MY_MPI_CUDA_FLAGS) -o test.o


NPROC := 8
NG := 8
BLOCKSIZE := 64
RUN := 1

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

bench:
	mpirun -n $(NPROC) nvprof --csv --log-file timings/run$(RUN)/new.%q{PMI_RANK}.txt ./test.o -Ng ${NG} -blockSize $(BLOCKSIZE)