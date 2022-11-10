demo:
	nvcc -c kernels/timer.cu -o timer.o
	nvcc -c kernels/a2a_ffts.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2a_ffts.o
	nvcc -c kernels/a2a_to_pencils.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o a2a_to_pencils.o
	nvcc -c kernels/change_fast_axis.cu -lineinfo -Xptxas -v -Xcompiler="-fPIC" -arch=sm_53 -shared -o change_fast_axis.o
	mpicc -c all2all_fft_cu_demo.c -o all2all_fft_cu_demo.o
	mpicc -c initializer/initializer.c -o initializer.o
	mpicxx all2all_fft_cu_demo.o timer.o a2a_ffts.o a2a_to_pencils.o change_fast_axis.o initializer.o -L/usr/local/cuda-10.2/lib64 -lcudart -o main.o

NPROC := 8
NG := 8

run:
	mpirun -n $(NPROC) ./main.o $(NG)

stitch:
	python3 stitch.py

clean:
	rm *.o
	rm proc*
	rm params