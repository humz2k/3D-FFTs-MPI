import numpy as np
import matplotlib.pyplot as plt

folder = "/home/hqureshi/3D-FFTs-MPI/src/a2afft/"

def get_coords(rank,step,raw_data,Ng):
    raw = np.real(raw_data[rank][step]).astype(np.int32)
    xs = np.zeros_like(raw)
    ys = np.zeros_like(raw)
    zs = np.zeros_like(raw)
    for idx,i in enumerate(raw):
        x,y,z = gridID2xyz(i,Ng)
        xs[idx] = x
        ys[idx] = y
        zs[idx] = z
    return xs,ys,zs,raw

def plot(step,nproc,raw_data,ng):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for rank in range(nproc):
        x,y,z,raw = get_coords(rank,step,raw_data,ng)
        ax.plot(x,y,z)

    plt.savefig("test.jpg")
    plt.close()

def rank2coords(rank,dims):
    out = np.zeros(3,dtype=int)
    out[0] = rank / (dims[1]*dims[2]);
    out[1] = (rank - out[0]*(dims[1]*dims[2])) / dims[2];
    out[2] = rank - out[0]*(dims[1]*dims[2]) - out[1]*(dims[2])
    return out

def local_grid_start(local_grid,coords):
    return local_grid * coords

def stitch(raw_data,dims,ng,local_grid,step=0):
    out = np.zeros((ng,ng,ng),dtype=np.complex64)
    for rank in range(raw_data.shape[0]):
        offset = local_grid_start(local_grid,rank2coords(rank,dims))
        data = raw_data[rank][step].reshape(local_grid)
        for i in range(local_grid[0]):
            for j in range(local_grid[1]):
                for k in range(local_grid[2]):
                    out[i+offset[0],j+offset[1],k+offset[2]] = data[i,j,k]
    return out

def gridID2xyz(idx, Ng):

    out = np.zeros(3,dtype=int)

    out[0] = idx / (Ng*Ng);
    out[1] = (idx - out[0]*(Ng*Ng)) / Ng;
    out[2] = idx - out[0]*(Ng*Ng) - out[1]*(Ng)

    return out.tolist()

def check(printStats=False):
    params = np.fromfile(folder + "/dims.out",dtype=np.int32)

    nproc = params[0]
    ng = params[1]
    nlocal = params[2]
    dims = params[3:6]
    local_grid = params[6:]
    if printStats:
        print("nproc:",nproc)
        print("ng:",ng)
        print("nlocal:",nlocal)
        print("dims:",dims)
        print("local_grid:",local_grid)

    raw_data = []
    for proc in [folder + "/proc" + str(i) for i in range(nproc)]:
        raw = np.fromfile(proc,dtype=np.float64)
        data = raw[::2] + 1j * raw[1::2]
        data = data.reshape((data.shape[0] // nlocal,nlocal))
        raw_data.append(data)

    raw_data = np.array(raw_data)
    
    inp = stitch(raw_data,dims,ng,local_grid,0)

    out = stitch(raw_data,dims,ng,local_grid,1)

    inv = stitch(raw_data,dims,ng,local_grid,2)

    return np.allclose(np.fft.fftn(inp),out), np.allclose(inp,inv)

print(check())