import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lns

from matplotlib import cm
from mpl_toolkits.mplot3d import (Axes3D, art3d)


######### FUNCTIONS #########


### Compute Z (=L2) at the decision boundary
def compute_Z(t1,t2,tc,L1,N,m):

	Z = np.log10(-10**L1*(t1**N-t1**m*tc**(N-m))/(t2**N-t2**m*tc**(N-m)))
	
	# Don't plot negative number of ligands
	Z[Z<0] = 0

	# Show up to a maximum to enhance prettiness
	if N > 1:
		for idxr,row in enumerate(Z>5):
			index = np.where(row == False)[0][-1]
			if t2 < 1:
				Z[idxr,index+1:index+2] = 5
				Z[idxr,index+2:] = np.nan
			else:
				Z[idxr,index+1:index+15] = 5
				Z[idxr,index+15:] = np.nan
				
	return Z


### Draw the immune boundary for different pairs (N,m) and tau_2
def draw_boundary(X,Y,Z,ax,N,m,t1,t2,l1,l2,marker,color):
	
	if t2 < 1:
		ax.view_init(25,345)
	else:
		ax.view_init(30,330)

	if N is 1:
		ax.plot_surface(X, Y, Z, linewidth=1, vmin = 0, vmax = 4)
	else:
		ax.plot_surface(X, Y, Z, linewidth=1, vmin = 0, vmax = 6)

	coords = [(t1,l1,l2[0]),(t1,l1,l2[1])]
	line_coords = [[(t1,t1),(l1,l1),(0,l2[0])],[(t1,t1),(l1,4),(l2[0],l2[0])],[(t1,t1),(l1,l1),(l2[0],l2[1])],[(t1,t1),(l1,4),(l2[1],l2[1])]]

	for idx,(x,y,z) in enumerate(coords):
		ax.scatter3D(x,y,z,marker = marker[idx], s = 150,color = color, edgecolor = None)

	for idx,line in enumerate(line_coords):
		ax.add_line(art3d.Line3D(line[0],line[1],line[2], color = color, ls = "dashed"))

	ax.set_xticks([3,4,5,t1])
	ax.set_yticks([1,2,3])
	ax.set_zticks([1,2,3,4,5])

	xticks = ["3","","5",""]
	yticks = ["   $10^{1}$", "", "   $10^{3}$"]
	zticks = ["    $10^{1}$", "", "    $10^{3}$", "", "    $10^{5}$"]

	ax.set_xticklabels(xticks)
	ax.set_yticklabels(yticks)
	ax.set_zticklabels(zticks)

	ax.tick_params(direction='in', pad=-5)

	ax.xaxis.set_rotate_label(False) 
	ax.yaxis.set_rotate_label(False) 
	ax.zaxis.set_rotate_label(False)

	ax.set_xlabel('$\\tau_1$\n\n',fontsize = 10)
	ax.set_ylabel('$L_1$\n\n\n',fontsize = 10) # (agonist)
	ax.set_zlabel('$L_2$      ',fontsize =10) # (antagonist)

	ax.set_xlim3d(3,t1-0.1)
	ax.set_ylim3d(0.1,3.9)
	ax.set_zlim3d(0.1,5)


######### MAIN #########


def main():
	pairs = [(1,0),(5,3)]
	numMesh = 30

	t1 = 6 # 
	tau_2 = [0.1,3]
	lig_list = ["self","antagonist"]
	tc = 4 # Threshold
	tau_1 = np.linspace(4,t1,numMesh)
	L1 = np.linspace(0,4,numMesh)
	X, Y = np.meshgrid(tau_1, L1)

	l1 = [1,np.log10(500)]
	l2 = [np.log10(50),4]

	marker = [("X","X"),("o","o"),("o","X"),("o","o")]
	color = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
	color[0]=color[1]

	fig = plt.figure(figsize = (8,8))

	for idxt,t2 in enumerate(tau_2):

		for idxp,(N,m) in enumerate(pairs):

			idx = 2*idxt+idxp

			Z = compute_Z(X,t2,tc,Y,N,m)
			ax = fig.add_subplot(2,2,idx+1, projection = '3d')

			draw_boundary(X,Y,Z,ax,N,m,t1,t2,l1[idxt],l2,marker[idx],color[3-2*idxt-idxp])

			# Title for columns
			ax.set_title("(N,m) = (%d,%d); $\\tau_2$ = %.1f" % (N,m,t2),fontsize = 10)


	title = "Boundary tilting for immune recognition"
	filename = "Boundary_tilting_new.pdf"

	# plt.suptitle(title, y = 0.92)
	plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight', pad_inches = 0.4)
	plt.close()

if __name__ is "__main__":
	main()