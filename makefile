thing:
	mpicc test.c helpers/helpers.c -o test

NPROC := 8

run:
	mpirun -n $(NPROC) ./test