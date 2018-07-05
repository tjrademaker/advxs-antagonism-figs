from sklearn import (datasets, decomposition, model_selection, svm)
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lns

import os


######### FUNCTIONS #########


### Call and split MNIST in a dataset with num1 and num2
def prepareData(num1,num2):

    mnist = datasets.fetch_mldata("MNIST original")
    # Rescale the data and extract all images for two digits
    X, y = mnist.data, mnist.target

    # Normalization
    X = (X-X.mean())/X.std()

    index = np.where((mnist.target == num1) | (mnist.target == num2))[0]

    # For traditional training/test split
    # train_index, test_index = index[index < 60000], index[index >= 60000]
    # return X[train_index], X[test_index], y[train_index], y[test_index]

    # For random training/test split
    return model_selection.train_test_split(X[index], y[index], test_size=0.2)


### Visualize projection on first PCs
def visualizePCA(X1_PC, X2_PC, coords, ax, PC, lims, idx, eps, score):

    # Plot linear SVC
    ax.add_line(lns.Line2D(lims[1], lims[2]))
    
    # Plot digits
    ax.scatter(X1_PC[:,PC[0]], X1_PC[:,PC[1]],s = 0.5, color = 'deepskyblue')
    ax.scatter(X2_PC[:,PC[0]], X2_PC[:,PC[1]],s = 0.5, color = 'salmon')

    marker = ["*","*","v","v"]
    color = ["darkblue","darkred","darkblue","darkred"]

    # Plot markers
    for idxC,[(x,y)] in enumerate(coords):
        ax.scatter(x,y, s = 100, marker = marker[idxC], c = color[idxC])

    if idx[1] is 0:
        ax.set_title("$\epsilon_{\mathrm{train}}$ = %.1f\tscore = %.2f" % (eps[0],score),fontsize=15, y = 1)
    elif idx[1] is 1:
        ax.set_title("score = %.2f" % score, fontsize=15, y = 1)
    if idx[0] is 0:
        ax.set_ylabel("$\epsilon_{\mathrm{test}}$ = %.1f" % eps[1], fontsize=15)
    elif idx[0] is 1 and idx[1] is 1:
        ax.set_xlabel("$\mathregular{PC_%d}$" % (PC[0]+1),fontsize=15)
        ax.set_ylabel("$\mathregular{PC_%d}$" % (PC[1]+1),fontsize=15)

    ax.set_xticks([],[])
    ax.set_yticks([],[])
    ax.axis(np.asarray([lims[0], lims[2]]).reshape(-1))


### Plot image + adversarial example
def plot_digits(X,X_adv,digit):

    fig, axes = plt.subplots(2,1, figsize = (4,4))
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = np.min(X), np.max(X)

    # Compare average 3s and 7s w/o adversary
    for coef, ax in zip(np.column_stack([X.T,X_adv.T]).T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())

    filename = "%s_eps.pdf" % digit
    plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
    plt.close()


######### MAIN #########


def main():
    
    # Numbers to classify
    num1,num2 = (3,7)

    # Set parameters
    eps_train = np.array([0,0.4])
    eps_test = np.array([0,0.4])
    PC = np.array((1,2))-1 # PCs to compare; subtract one to account for Python arrays starting indexing from 0

    # Split the dataset
    X_train, X_test, y_train, y_test = prepareData(num1,num2)

    # Compute difference between the averages of num1 and num2
    ave1 = X_test[y_test == num1].mean(axis=0)
    ave2 = X_test[y_test == num2].mean(axis=0)
    delta_train = X_train[y_train == num1].mean(axis=0) - X_train[y_train == num2].mean(axis=0)
    delta_test = ave1 - ave2

    # Plot some sample digits
    plot_digits(X_test[0], X_test[0] - 0.6*np.sign(delta_test), str(num1))
    plot_digits(X_test[-1], X_test[-1] + 0.6*np.sign(delta_test), str(num2))
    plot_digits(ave1,ave1 - 0.4*np.sign(delta_test), "ave" + str(num1))
    plot_digits(ave2,ave2 + 0.4*np.sign(delta_test), "ave" + str(num2))

    fig = plt.figure(figsize = (10,7))
    title = "Boundary tilting for %ss vs. %ss" % (str(num1),str(num2))
    filename = "Boundary_tilting_in_digits.pdf"

    # Loop through all eps_train and eps_test
    for idx1,eps1 in enumerate(eps_train):

        # Create arrays with adversarial perturbation for training and test set
        X_train_adv = np.zeros(X_train.shape)
        X_test_adv = np.zeros(X_test.shape)
        X_train_adv[ y_train == num1 ] = X_train[ y_train == num1 ] - eps1*np.sign( delta_train )
        X_train_adv[ y_train == num2 ] = X_train[ y_train == num2 ] + eps1*np.sign( delta_train )
        X_test_adv[y_test == num1] = X_test[y_test == num1] - eps_test.max()*np.sign(delta_test)
        X_test_adv[y_test == num2] = X_test[y_test == num2] + eps_test.max()*np.sign(delta_test)
        
        # Compute linear transformation by fitting on the regular images and adversarial examples combined
        pca = decomposition.PCA().fit(np.concatenate((X_train,X_train_adv),axis=0))
        
        # Find projection of train and test set on chosen PCs. In principal could project on any (number of) principal component(s)
        X_train_adv_pca = pca.transform(X_train_adv)[:,PC]
        X_test_pca = pca.transform(X_test)[:,PC]
        X_test_pca_adv = pca.transform(X_test_adv)[:,PC]

        # Compute in PC space the coordinates of the markers of single 3, 7, ave3, ave7, delta w/o advs.
        ave1_pca = pca.transform(ave1.reshape(1,-1))[:,PC]
        ave2_pca = pca.transform(ave2.reshape(1,-1))[:,PC]

        # Train Linear Support Vector Classifier on PCs (i.e. find line that best separates both classes)
        supvec = svm.LinearSVC()
        supvec.fit(X_train_adv_pca,y_train)

        # Compute x,y limits of plot and line that separates both classes
        xlims,ylims,xlims0,ylims0,xlims_eps,ylims_eps = [np.ndarray((1,2)).flatten()]*6
        xlims0 = np.asarray([X_test_pca[:,PC[0]].min(),X_test_pca[:,PC[0]].max()])
        ylims0 = np.asarray([X_test_pca[:,PC[1]].min(),X_test_pca[:,PC[1]].max()])
        xlims_eps = np.asarray([X_test_pca_adv[:,PC[0]].min(),X_test_pca_adv[:,PC[0]].max()])
        ylims_eps = np.asarray([X_test_pca_adv[:,PC[1]].min(),X_test_pca_adv[:,PC[1]].max()])
        
        xlims = np.asarray([np.min([xlims0,xlims_eps]),np.max([xlims0,xlims_eps])])
        ylims = np.asarray([np.min([ylims0,ylims_eps]),np.max([ylims0,ylims_eps])])
        xlims_line = -supvec.coef_[0][1]/supvec.coef_[0][0]*ylims-supvec.intercept_[0]/supvec.coef_[0][0]        
    
        # Now test how well the unperturbed and the perturbed testset are classified
        for idx2,eps2 in enumerate(eps_test):

            ax = fig.add_subplot(2,2,idx1+2*idx2+1)
            coords = [ave1_pca,ave2_pca]

            # Naive test set
            if idx2 is 0:
                X_test_adv_pca = pca.transform(X_test)[:,PC]

            # Perturbed test set
            elif idx2 is 1:
                X_test_adv = np.zeros(X_test.shape)
                X_test_adv[y_test == num1] = X_test[y_test == num1] - eps2*np.sign(delta_test)
                X_test_adv[y_test == num2] = X_test[y_test == num2] + eps2*np.sign(delta_test)
                X_test_adv_pca = pca.transform(X_test_adv)[:,PC]

                # Add ave1/ave2 shifted with eps*sgn(delta)
                coords.append(pca.transform((ave1 - eps2*np.sign(delta_test)).reshape(1,-1))[:,PC])
                coords.append(pca.transform((ave2 + eps2*np.sign(delta_test)).reshape(1,-1))[:,PC])

            score = supvec.score(X_test_adv_pca,y_test)
            visualizePCA(X_test_adv_pca[y_test == num1], X_test_adv_pca[y_test == num2],coords,ax,PC,(xlims,xlims_line,ylims),(idx1,idx2),(eps1,eps2),score)

    plt.suptitle(title, y = 0.975, fontsize = 20)
    plt.savefig(os.getcwd() + "/" + filename, dpi=300, bbox_inches = 'tight')
    plt.close()

if __name__ is "__main__":
    main()