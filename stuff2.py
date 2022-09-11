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

temp = procs[0]["step1"].reshape(((Ng*Ng)//n_procs),Ng,3)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in temp:

    color = (i[:,0][0]/7,1-i[:,1][0]/7,0)
    print(color)

    ax.plot(i[:,0],i[:,1],i[:,2],color=color)
plt.show()