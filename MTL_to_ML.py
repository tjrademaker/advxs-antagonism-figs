import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io


######### FUNCTIONS #########


### Compute T_N
def TN_dist(t,tc,N,m,L):
	return t**N@L/(t**m@L)/tc**(N-m)


### Compute gradient of T_N along all tau
def grad_TN(t,tc,N,m,L):
	return ((t**m@L)*N*t**(N-1)*L-(t**N@L)*m*t**(m-1)*L)/(t**m@L)**2/tc**(N-m)


### Setup ligand distribution
def setup_distribution(tc,Self,Anta,Ag,binsize):

	# Sample three normal distribution with (\mu,\sigma) = 2*(0,1),1*(0,0.1)
	t1_dist = np.random.normal(0,0.01,size = Ag) # Narrowly distributed
	t2_dist = -abs(np.random.normal(0,1/3,size = Anta)) # Just below tc
	t3_dist = abs(np.random.normal(0,1/3,size = Self)) # Just above 0

	# Combine three vectors into initial ligand distribution tau_init
	tau_init = np.sort(np.concatenate((tc+1/3+t1_dist,tc+t2_dist,t3_dist)))

	# Return bins and centered values of each bin
	t1 = int(tau_init.max()/binsize+1)*binsize
	tau_bins = np.linspace(0, t1, int(t1/binsize)+1)
	tau_center = np.linspace(binsize/2,t1-binsize/2,int(t1/binsize))

	return tau_init,tau_bins,tau_center


### Compute decision boundary
def find_boundary(tau,tau_bins,tau_center,tc,N,m,Self,Anta,Ag,epsilon,binsize):

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

		# Add correction to individual taus. There are as many corrections as bins
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
	tau_list = [tau[Self+Anta:],tau[Self:Self+Anta],tau[:Self]]
	min_tau, mean_tau, max_tau = [],[],[]

	for item in tau_list:
		min_tau.append(np.min(item))
		mean_tau.append(np.mean(item))
		max_tau.append(np.max(item))
	measures = tuple([item for sublist in [min_tau,mean_tau,max_tau] for item in sublist])
	
	print("\nRange of tau_1, tau_2 and tau_3")
	print("\ttau_1 \t tau_2 \t tau_3 \nmin \t%.2f \t %.2f \t %.2f \nmean \t%.2f \t %.2f \t %.2f \nmax \t%.2f \t %.2f \t %.2f" % measures)
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
		ax.set_xlabel("%d" % numIter, labelpad = 0.5, fontsize = 6)	


### Draw immune picture
def make_immune_picture(tau_init,tau_order,tc,M,ax):

	new_image = np.zeros(M).reshape(-1)

	# Set ordering
	for tau,i in zip(tau_init,tau_order):
		new_image[i] = tau
	new_image = new_image.reshape(M)

	p=ax.pcolormesh(np.flipud(new_image),cmap="Greys_r",vmin=0,vmax=tc)

	ax.set_xticks([])
	ax.set_yticks([])


### Draw initial distribution
def visualize_initial_distribution(ta,tc,tau_order,Self,Anta,Ag,M,binsize = 0.2):

	tau_init,tau_bins,tau_center = setup_distribution(tc,Self,Anta,Ag,binsize)
	L_init = np.histogram(tau_init,tau_bins)[0]

	fig,ax = plt.subplots(2,1,figsize=(2,2))

	plot_distribution(tau_center,L_init,ta,tc,Ag,np.prod(M),0,ax[0],binsize)
	make_immune_picture(tau_init,tau_order,tc,M,ax[1])

	filename = "initial_dist_MTL.png"

	plt.suptitle("Initial distribution")
	plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
	plt.close()


### Preprocess image
def read_image(colors):

	image = io.imread(os.getcwd() + "/MTL_before.png", as_grey = True)

	# All rows and columns with ambiguous pixel values
	# Include some rows at top/bottom to equalize self vs not self distribution
	rows = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,45,90,134,178,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224)
	cols = (17,61,105,150,166,178,223,266,312,325,339,384,426)

	image = np.delete(image, rows, axis = 0)
	image = np.delete(image, cols, axis = 1)

	# The diagonal on the M, which remains ugly, should be a regular stair, so slope 1, instead of 5/4
	image[:,60:105] = image[:,60:105].round()
	image = 255*image

	# Pixelvalues of black, darkgray, lightgray and white
	bl, dg, lg, wh = colors

	for i in range(len(image)):
		for j in range(len(image[0])):
			if image[i,j] == 0:
				image[i,j] = bl
			elif image[i,j] == 60:
				image[i,j] = dg
			elif image[i,j] == 220:
				image[i,j] = lg
			elif image[i,j] == 255:
				image[i,j] = wh

	with open('MTL.pickle', 'wb') as f:
		pickle.dump(image,f)

	fig = plt.figure()
	fig.tight_layout(pad = 0)
	plt.imshow(image,cmap = "Greys_r")
	plt.axis("off")
	plt.savefig(os.getcwd() + "/MTL_after.png")
	plt.close()

	return image


############ MAIN ####################


def main():

	filename = 'MTL.pickle'

	if os.path.exists(filename):
		with open(filename, 'rb') as f:
			image = pickle.load(f)
	else:
		colors = (0,10,15,25)
		image = read_image(colors)

	M = image.shape

	# Set parameters
	R = np.prod(M) # Number of receptors
	Ag = np.sum(image == 25) # Number of agonists 
	Anta = np.sum(image == 15) # Number of antagonists
	Self = int(R - Ag - Anta) # Number of self
	ta = 1 # Antagonists
	tc = 3 # Threshold

	int_step = 1 # For noninteger (N,m), int_step < 1
	numN = 4 # Max number of KPR steps
	numm = 4 # Max number of activation + 1

	binsize = 1/5 # Binsize of tau's in the histogram
	epsilon = 1/5 # Epsilon in tau = tau + eps*dtau

	# Looping variables
	rangeKPR = np.arange(numN/int_step+1)
	rangem = np.arange(numm/int_step+1)

	# Set ordering for all ligands
	order = []
	for pixel_val in np.unique(image):
		order.append(np.where(image.reshape(-1) == pixel_val)[0])
	tau_order = np.concatenate(order)

	# Draw initial ligand distribution
	visualize_initial_distribution(ta,tc,tau_order,Self,Anta,Ag,M,binsize)

	fig, axes = plt.subplots(len(rangem),len(rangeKPR), figsize = (6,3))

	filename = "MTL_to_ML.png"

	# Loop through the all configurations (N,m)
	for axX in rangeKPR:

		for axY in rangem:						

			# Subplots on the diagonal remain empty
			if int(axX) == int(axY):
				ax = axes[int(axX),int(axY)]
				ax.axis("off")
				continue

			# Setup ligand distribution
			tau_init,tau_bins,tau_center = setup_distribution(tc,Self,Anta,Ag,binsize)

			# Ligand distribution in lower diagonal, immune pictures in upper diagonal
			if axY > axX:
				N = axY + 1; m = axX + 1
				print("\n(N,m) = (%d,%.d)" % (N,m))

				tau_final, L_final, numIter = find_boundary(tau_init,tau_bins,tau_center,tc,N,m,Self,Anta,Ag,epsilon,binsize)
				
				ax = axes[int(axX),int(axY)]
				make_immune_picture(tau_final,tau_order,tc,M,ax)

				ax = axes[int(axY),int(axX)]
				plot_distribution(tau_center,L_final,ta,tc,Ag,R,numIter,ax,binsize)

			else:
				N = axX + 1; m = axY + 1
				ax = axes[int(axY),int(axX)]
			
			# Axes labels
			if int(axX) is 0:
				ax.set_ylabel("N = %d" % N,fontsize = 8, labelpad = 9)
			if int(axY) is 0:
				ax.set_title("N = %d" % N, fontsize = 8, y = 1.1)
			if int(axX) == int(axY-1):

				if m < 2:
					color_m = "red"
				if m == 2:
					color_m = "black"
				if m > 2:
					color_m = "green"

				ax.set_title("m = %d" % m, y = 1.4, fontsize = 8, fontweight = 'bold', color = color_m)

	plt.subplots_adjust(hspace = .3)
	plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
	plt.close()


if __name__ == "__main__":
	main()