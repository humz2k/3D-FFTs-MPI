import numpy as np
import matplotlib.pyplot as plt

n_procs = 8
Ng = 8
n_grid_points = Ng*Ng*Ng

my_grid_points = int(n_grid_points / n_procs)

print(my_grid_points)

procs = []

for i in range(n_procs):
    procs.append({})
    filename = "proc" + str(i)
    with open(filename,"r") as f:
        lines = f.read().splitlines()
    
    procs[i]["id"] = lines[0]
    lines = lines[1:]

    while len(lines) > 0:
        step = lines[1:][:my_grid_points]

        procs[i][lines[0]] = np.array([[int(j) for j in i.split(",")] for i in step])

        lines = lines[1:][my_grid_points:]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for idx,proc in enumerate(procs):
    coords = proc["step2"]
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]

    ax.scatter(xs,ys,zs,label=str(idx))

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.legend()

plt.show()

    