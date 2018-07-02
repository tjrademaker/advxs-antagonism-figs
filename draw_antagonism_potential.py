import os

import numpy as np
import matplotlib.pyplot as plt


######### FUNCTIONS #########


### Compute T_N
def TN(t1,t2,L1,L2,tc,N,m):
	summ = (t1**m*L1+t2**m*L2)
	sumN = (t1**N*L1+t2**N*L2)
	return sumN/summ/tc**(N-m)


### Compute gradient of T_N w.r.t. t2
def grad_TN(t1,t2,L1,L2,tc,N,m):
	summ = (t1**m*L1+t2**m*L2)
	dsumm = m*t2**(m-1)*L2
	sumN = (t1**N*L1+t2**N*L2)
	dsumN = N*t2**(N-1)*L2
	return (summ*dsumN-sumN*dsumm)/summ**2/tc**(N-m)


### Compute gradient squared of T_N w.r.t. t2
def grad2_TN(t1,t2,L1,L2,tc,N,m):
	summ = (t1**m*L1+t2**m*L2)
	dsumm = m*t2**(m-1)*L2
	ddsumm = m*(m-1)*t2**(m-2)*L2
	sumN = (t1**N*L1+t2**N*L2)
	dsumN = N*t2**(N-1)*L2
	ddsumN = N*(N-1)*t2**(N-2)*L2

	return (summ**2*(summ*ddsumN - sumN*ddsumm) - 2*dsumm*summ*(summ*dsumN-sumN*dsumm))/summ**4/tc**(N-m)


### Compute minimum of a function
def root_finding(a):
	root = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
	if (sum(root) == 0) or (root[0] == True):
		return 0
	else:
		return root


############ MAIN ####################


def main():

	# Initialize parameters
	R = int(1e4) # Number of receptors
	L1 = int(.4*R) # Number of agonists
	L2 = R-L1 # Number of self
	tc = 3 # Threshold
	t1 = 3
	t2 = np.arange(0.01,tc,0.01)

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']

	# Looping variables
	rangeKPR = np.arange(1,6)
	rangem = np.arange(0,5)

	fig,axes = plt.subplots(len(rangeKPR),1,figsize=(2,8))

	title = "Antagonism potential"
	filename = "Antagonism_potential.pdf"

	# Loop through the all configurations (N,m)
	for idxN,N in enumerate(rangeKPR):
		
		legend_label = []
		ax = axes[idxN]

		for m in rangem:

			if int(m) >= int(N):
				continue

			# Plot T_N as a function of tau_2
			p = ax.plot(t2,TN(t1,t2,L1,L2,tc,N,m), color = colors[m+3],label = m)
			legend_label.append("%.1f"%m)

			# Find minima and inflexion points
			idx_min = root_finding(TN(t1,t2,L1,L2,tc,N,m))
			idx_in = root_finding(grad_TN(t1,t2,L1,L2,tc,N,m))

			if idx_min is not 0:
				t2_min = t2[idx_min][0]
				ax.plot(t2_min,TN(t1,t2_min,L1,L2,tc,N,m), "o", color = colors[m+3], markersize = 3)
			if idx_in is not 0:
				t2_in = t2[idx_in][0]
				ax.plot(t2_in,TN(t1,t2_in,L1,L2,tc,N,m),"s", color = colors[m+3], markersize = 3)

		ax.set_ylim([0.38,1.02])
		ax.set_xlim([-0.1,tc])
		ax.set_yticks([])
		ax.set_xticks([0,1,2,3])
		ax.set_xticklabels(['','','',''])

		ax.set_ylabel("$T_{%d,m}(\\tau_2)$" % N)

		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles[::-1], labels[::-1], title='m', loc='lower right', markerscale = 0.1, prop={'size': 5})

	axes[-1].set_xlabel("$\\tau_2$", labelpad = -5)
	axes[-1].set_xticklabels(['0','','','$\\tau_c$'])

	plt.suptitle(title, y = 0.91)#, horizontalalignment = "left")
	plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
	plt.close()


if __name__ is "__main__":
	main()