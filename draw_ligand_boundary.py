import os
import numpy as np
import matplotlib.pyplot as plt


######### FUNCTIONS #########


### Compute T_N
def TN_dist(t,tc,N,m,L):
	return t**N@L/(t**m@L)/tc**(N-m)


### Compute gradient of T_N along all tau
def grad_TN(t,tc,N,m,L):
	return ((t**m@L)*N*t**(N-1)*L-(t**N@L)*m*t**(m-1)*L)/(t**m@L)**2/tc**(N-m)


### Setup ligand distribution
def setup_distribution(t1,s_t1,t2,s_t2,Ag,R,binsize):

	# Sample two normal distribution with (\mu,\sigma) = (0,s_ti)
	t1_dist = np.random.normal(0,s_t1,size = Ag)
	t2_dist = abs(np.random.normal(0,s_t2,size = R-Ag)) # Take absolute value of tau_2 to avoid negative binding times because tau_2 is small

	# Combine two vectors into initial ligand distribution tau_init
	tau_init = np.sort(np.concatenate((t1+t1_dist,t2+t2_dist)))

	# Return bins and the centered values of each bin
	tau_bins = np.linspace(0, t1, int(t1/binsize)+1)
	tau_center = np.linspace(binsize/2,t1-binsize/2,int(t1/binsize))

	return tau_init,tau_bins,tau_center


### Compute decision boundary
def find_boundary(tau,tau_bins,tau_center,tc,N,m,Ag,R,epsilon,binsize):

	# Make a histogram of all binding times tau via the tau_bins in L
	L = np.histogram(tau,tau_bins)[0]

	# Initialize dtau and TNs
	dtau = np.zeros(tau.shape)
	TN = TN_dist(tau_center,tc,N,m,L)*np.ones((int(1e6),1))
	
	# Loop until current T_N < 1 (= no response) or until max number of iterations has been reached (= 1e6)
	i = 1
	while ((TN[i] > 1) & (i < len(TN)-1)):

		# Compute T_N and grad[T_N] in terms of binned taus
		TN[i+1] = TN_dist(tau_center,tc,N,m,L)
		dtau_bin = grad_TN(tau_center,tc,N,m,L)

		# Add correction to individual taus. There are as many corrections as bins under tau_c
		itemcount = 0 # This tracks the ligands in each bin in L
		for idx,item in enumerate(L[:int(tc/binsize)]): 

			dtau[itemcount:itemcount+item] = dtau_bin[idx]
			itemcount += item

		tau -= epsilon*dtau

		# If tau_2 becomes smaller than 0, set it back to smallest value
		tau[tau < 0] = tau_center[0]
		tau.sort()

		# Rebin taus
		L = np.histogram(tau,tau_bins)[0]

		# Move to next iteration
		i += 1

	# Once converged, print results
	tau_list = [tau[R-Ag:],tau[:R-Ag]]
	min_tau, mean_tau, max_tau = [],[],[]

	for item in tau_list:
		min_tau.append(np.min(item))
		mean_tau.append(np.mean(item))
		max_tau.append(np.max(item))
	measures = tuple([item for sublist in [min_tau,mean_tau,max_tau] for item in sublist])
	
	print("\nRange of tau_1 and tau_2")
	print("\ttau_1 \t tau_2 \nmin \t%.2f \t %.2f \nmean \t%.2f \t %.2f\nmax \t%.2f \t %.2f" % measures)
	print("\nLigand distribution across bins")
	print(L[:int(tc/binsize)+10])
	print("\nT_N = %.10f\t Num_iter = %d\n" % (TN[i], i)) # Print T_N and number of iterations
	
	return tau, L, i


### Plot ligand distribution at boundary
def plot_distribution(tau,L,ta,tc,Ag,R,numIter,ax,binsize):
	
	bool_ta = (tau <= ta)
	bool_ta_tc = (tau > ta) & (tau < tc)
	bool_tc = (tau >= tc)

	ax.bar(tau[bool_tc],L[bool_tc], width = binsize, label = "agonist")
	ax.bar(tau[bool_ta_tc],L[bool_ta_tc], width = binsize, label = "antagonist")
	ax.bar(tau[bool_ta],L[bool_ta], width = binsize, label = "self")	
	ax.axvline(tc, color='grey', linestyle='dashed', linewidth=0.5, label = "$\\tau_c$")
	
	ax.set_xticks([])
	ax.set_yticks([])

	ax.set_xlim([0,int(tau.max()*5+1)/5])
	ax.set_ylim([0,R-Ag])

	# Write number of iterations as xlabel
	if numIter is not 0:
		ax.set_xlabel("%d" % numIter, fontsize = 8, labelpad = 0.5)


### Draw immune picture
def make_immune_picture(tau,tau_order,tc,R,ax):

	# Set ordering
	tau = tau[tau_order].reshape(int(np.sqrt(R)),int(np.sqrt(R)))

	p=ax.pcolormesh(tau,cmap="Greys_r",vmin=0,vmax=tc)

	ax.set_xticks([])
	ax.set_yticks([])


### Draw initial distribution
def visualize_initial_distribution(t1,s_t1,t2,s_t2,ta,tc,tau_order,Ag,R,binsize = 0.2):
	
	tau_init,tau_bins,tau_center = setup_distribution(t1,s_t1,t2,s_t2,Ag,R,binsize)
	L_init = np.histogram(tau_init,tau_bins)[0]

	fig,ax = plt.subplots(1,2,figsize=(2.15,1))

	plot_distribution(tau_center,L_init,ta,tc,Ag,R,0,ax[0],binsize)
	make_immune_picture(tau_init,tau_order,tc,R,ax[1])

	filename = "initial_dist_pic.png"

	plt.suptitle("Initial distribution", y = 1.05)
	plt.savefig(os.getcwd() + "/"  + filename, dpi=300, bbox_inches = 'tight')
	plt.close()


############ MAIN ####################


def main():

	# Initialize parameters
	R = int(np.sqrt(1e4)+1)**2 # Number of receptors 
	Ag = int(.3*R) # Ag = number of agonists; R - Ag = number of self
	t1 = 7/2 # mean tau_1
	s_t1 = 1/10 # variance in tau_1
	t2 = 0 # min tau_2
	s_t2 = 1/3 # variance in tau_2
	ta = 1 # Antagonists
	tc = 3 # Threshold

	int_step = 1 # For noninteger (N,m), int_step < 1
	numN = 5 # Max number of KPR steps
	numm = 5 # Max number of activation + 1

	binsize = 1/5 # Binsize of tau's in the histogram
	epsilon = 1/5 # Epsilon in tau = tau + eps*dtau

	# Looping variables
	rangeKPR = np.arange(numN/int_step+1)
	rangem = np.arange(numm/int_step+1)

	# Set ordering for all ligands
	tau_order = np.random.permutation(R)

	# Draw initial ligand distribution
	visualize_initial_distribution(t1,s_t1,t2,s_t2,ta,tc,tau_order,Ag,R,binsize)

	fig, axes = plt.subplots(len(rangem),len(rangeKPR), figsize = (6,6))

	title = "Ligand distribution at the boundary"
	filename = "Ag_fraction_%.1f_t1_%.1f_%.2f_t2_%.1f_%.2f.png" % (Ag/R,t1,s_t1,t2,s_t2)

	# Loop through the all configurations (N,m)
	for axX in rangeKPR:

		for axY in rangem:						

			# Subplots on the diagonal remain empty
			if int(axX) == int(axY):
				ax = axes[int(axX),int(axY)]
				ax.axis("off")
				continue

			# Setup ligand distribution
			tau_init,tau_bins,tau_center = setup_distribution(t1,s_t1,t2,s_t2,Ag,R,binsize)

			# Ligand distribution in lower diagonal, immune pictures in upper diagonal
			if axY > axX:
				N = axY; m = axX
				print("\n(N,m) = (%d,%.d)" % (N,m))

				tau_final, L_final, numIter = find_boundary(tau_init,tau_bins,tau_center,tc,N,m,int(Ag),R,epsilon,binsize)
				
				ax = axes[int(axX),int(axY)]
				make_immune_picture(tau_final,tau_order,tc,R,ax)

				ax = axes[int(axY),int(axX)]
				plot_distribution(tau_center,L_final,ta,tc,Ag,R,numIter,ax,binsize)

			else:
				N = axX; m = axY
				ax = axes[int(axY),int(axX)]
			
			# Axes labels
			if int(axX) is 0:
				ax.set_ylabel("N = %d" % N,fontsize = 12, labelpad = 11)
			if int(axY) is 0:
				ax.set_title("N = %d" % N, fontsize = 12, y = 1.2)
			if int(axX) == int(axY-1):

				if m < 2:
					color_m = "red"
				if m == 2:
					color_m = "black"
				if m > 2:
					color_m = "green"

				ax.set_title("m = %d" % m, y = 1.4, fontsize = 12, fontweight = 'bold', color = color_m)

	plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
	plt.close()


if __name__ is "__main__":
	main()