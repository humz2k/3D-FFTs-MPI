import os
import stitch
import numpy as np

def run_test(world_size=8,Ng=8,blockSize=64):

    print("Test: world_size = " + str(world_size) + ", Ng = " + str(Ng) + ", blockSize = " + str(blockSize))

    os.system("make run NPROC=" + str(world_size) + " NG=" + str(Ng) + " BLOCKSIZE=" + str(blockSize) + " > test_" + str(world_size) + "_" + str(Ng) + "_" + str(blockSize) + ".log")

    forward,inverse = stitch.check()

    print("     Forward:",["FAILED","PASSED"][forward])
    print("     Inverse:",["FAILED","PASSED"][inverse])

    os.system("make clear_test > test_" + str(world_size) + "_" + str(Ng) + "_" + str(blockSize) + ".log")

run_test(world_size=8,Ng=8,blockSize=64)
run_test(world_size=8,Ng=32,blockSize=128)
run_test(world_size=6,Ng=12,blockSize=128)
run_test(world_size=4,Ng=4,blockSize=64)