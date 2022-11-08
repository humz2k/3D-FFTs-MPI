demo:
	mpicc all2all_fft_pencils_demo.c helpers/helpers.c a2a_to_pencils/a2a_to_pencils.c change_fast_axis/change_fast_axis.c -o demo

thing:
	mpicc test.c helpers/helpers.c -o test

NPROC := 8

run:
	mpirun -n $(NPROC) ./demo