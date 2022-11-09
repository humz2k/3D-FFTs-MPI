demo:
	nvcc -c kernels/*.cu -o kernels.o
	mpicc -c all2all_fft_cu_demo.c -o all2all_fft_cu_demo.o
	mpicc all2all_fft_cu_demo.o kernels.o -L/usr/local/cuda-10.2/lib64 -lcudart -o main.o

run:
	./main.o