thing:
	mpicc test.c -o test

NPROC := 8

run:
	mpirun -n $(NPROC) ./test