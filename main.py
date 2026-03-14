# For video processing and is deprecated, 
# but only experimental Windows support for torchdecoders
# See: https://github.com/pytorch/torchcodec
# Last Statis: 25th September 2025

# To suppress deprecation error
import warnings ; warnings.warn = lambda *args, **kwargs: None
#warnings.filterwarnings("ignore", category=DeprecationWarning)

import math
import numpy as np

# For tensors
import torch
# For I/O, but deprecated, no alterantive fully supported for Windows as seen above
from torchvision.io import VideoReader
from torchvision.transforms import v2


# Algorithm class
from algorithm.TensorADMM import TensorADMM

# Detector class
from detection.Detection import Detection

# For parallelization of different algorithm instances
from joblib import Parallel, delayed

# For plotting
import matplotlib.pyplot as plt


# Runs the algorithm instance (object of class TensorADMM) with niter iterations
def run(instance, niter):
    instance.algorithm_and_measurements(niter)
    return instance

# Plotting helper method
def plot(tensor_admm, niter):
    # For plotting, three different time intervals:
    #   1. Including starting iterates 
    #   2. From first iterate
    #   3. Cut-Off before {t_cutoff}-th iterates
    # time axis
    interval_total = np.linspace(0, niter, niter+1)
    interval_after_1 = np.linspace(1, niter, niter)

    # Cut-Off value: We only plot values after skipping {t_cutoff} points, t_cutoff < niter
    t_cutoff = 50
    interval_after_cutoff = np.linspace(t_cutoff, niter, niter-t_cutoff+1)

    # Augmented Lagrangian and after Cut-Off
    fig, axs = plt.subplots(layout = "constrained")
    # Augmented Lagrangian valus
    for i in range(len(beta)): 
        axs.plot(interval_total, tensor_admm[i].logging["values_aug_lagr"], label=f"$\\beta = {beta[i]}$")
        axs.legend(loc='upper right')
    axs.set(ylabel="$L_{\\beta}(X^{k}, Y^{k}, Z^{k}, W^{k})$")
    fig.savefig('./plot_statistics/AugmentedLagrangianValues.png')

    # Augmented Lagrangian values after skipping t_cutoff iterates
    fig, axs = plt.subplots(layout = "constrained")
    for i in range(len(beta)): 
        axs.plot(interval_after_cutoff, tensor_admm[i].logging["values_aug_lagr"][t_cutoff:], label=f"$\\beta = {beta[i]}$")
        axs.legend(loc='upper right')
    axs.set(ylabel="$L_{\\beta}(X^{k}, Y^{k}, Z^{k}, W^{k})$")
    fig.savefig('./plot_statistics/AugmentedLagrangianValuesAfter50.png')

    # Augmented Lagrangian values for beta = 10, 20 after skipping t_cutoff iterates
    fig, axs = plt.subplots(layout = "constrained")
    for i in range(2): 
        axs.plot(interval_after_cutoff, tensor_admm[i].logging["values_aug_lagr"][t_cutoff:], label=f"$\\beta = {beta[i]}$")
        axs.legend(loc='upper right')
    axs.set(ylabel="$L_{\\beta}(X^{k}, Y^{k}, Z^{k}, W^{k})$")
    fig.savefig('./plot_statistics/AugmentedLagrangianValuesAfter50_beta1020.png')

    # Feasibility Violation + Objective
    # Objective
    fig, axs = plt.subplots(layout = "constrained")
    for i in range(len(beta)): 
        axs.plot(interval_total, tensor_admm[i].logging["objective"], label=f"$\\beta = {beta[i]}$")
    axs.legend(loc = 'upper right')
    axs.set(ylabel="$\\phi(X^{k}, Y^{k}, Z^{k})$")
    fig.savefig('./plot_statistics/objective.png')
    
    # Feasibility Violation
    fig, axs = plt.subplots(layout = "constrained")
    for i in range(len(beta)): 
        axs.plot(interval_total, tensor_admm[i].logging["values_feas"], label=f"$\\beta = {beta[i]}$")
    axs.legend(loc = 'upper right')
    axs.set(ylabel="$ \\Vert X^{k} + Y^{k} + Z^{k} - M \\Vert $")
    fig.savefig('./plot_statistics/feasibility.png')
    
    
    # Descent and lower bound
    # Descent in augmented Lagrangian values + Theoretical lower bound
    fig, axs = plt.subplots(2,2, layout = "constrained")
    for i in range(2): 
        for j in range(2):
            index = i*2 + j
            axs[i,j].plot(interval_after_1, tensor_admm[index].logging["values_descent"], 'b', label=f"$\\beta = {beta[index]}$")
            axs[i,j].plot(interval_after_1, tensor_admm[index].logging["values_descent_lower"], 'r', label=f'Lower Bound for $\\beta = {beta[index]}$')
            axs[i,j].legend(loc = 'upper right')
    fig.savefig('./plot_statistics/descent_lower_bound.png')

    # Subgradient norm of augmented Lagrangian and theoretical upper bound
    fig, axs = plt.subplots(2,2, layout = "constrained")
    for i in range(2): 
        for j in range(2):
            index = i*2 + j
            axs[i,j].plot(interval_after_1, tensor_admm[index].logging["values_subgr_upper"], 'r',
                           label=f"Upper Bound for $\\beta = {beta[index]}$")
            axs[i,j].plot(interval_after_1, tensor_admm[index].logging["values_subgr"], 'b',
                           label=f'$\\beta = {beta[index]}$')
            axs[i,j].legend(loc = 'upper right')
    fig.savefig('./plot_statistics/subgradient_upper_bound.png')

# If wanted, summary statistics
def summary_statistics(tensor_admm):
    for i in range(len(beta)):
        print(f"For beta={beta[i]}: ")

        print(f"Current augmented Lagrangian value: {tensor_admm[i].logging["values_aug_lagr"][-1]}")
        print(f"Current augmented Lagrangian descent: {tensor_admm[i].logging["values_descent"][-1]}")
        print(f"Current subgradient norm: {tensor_admm[i].logging["values_subgr"][-1]}")
        
        print()

# Detection
def detect(beta, iterations):
    # Initialization
    object_admm = list()
    # New instances of class TensorADMM with same hyperparameters
    for i in range(len(iterations)):
        object_admm.append(TensorADMM(video, X_0, Y_0, Z_0, param = (beta, q, C_x, lambd, mu)))


    # Detector
    # Color red + Instantiation of detector
    red = (255,0,0)
    color = red
    
    detector = Detection(video, color)
    
    # Run both instances
    # Execute the ADMM instances
    for i in range(len(iterations)):
        object_admm[i] = run(object_admm[i], iterations[i])
        detector.paint_detected_area(object_admm[i].Y_, f'{iterations[i]}_iters')
        
    

if __name__ == '__main__':

    

    "GET VIDEO IN TENSOR FORMAT WITH GRAYSCALE VALUE"

    video_path = "./data/GrayscaleParking.mp4"

    # Define sizes of video with time_steps many m x n - frames
    #h = int(2160 / 12)
    #w = int(3840 / 24)
    #time_steps = 32

    reader = VideoReader(video_path)

    # Storage
    frames = torch.Tensor() # torch.Tensor(size = (time_steps, 3, h, w))

    # Read frames into tensor

    for frame in reader:
        frames.append(frame['data'])

    # Store as grayscale tensor, since original video was grayscale
    video = v2.Grayscale(1)(frames)[:,0,:,:]

    time_steps, h, w = video.shape

    "DEFINE INITIAL ITERATES AND HYPERPARAMETERS (as in thesis)"

    # Low rate of change between frames, hence take the frames, average, and then set each frame as this average
    X_0 = (torch.sum(video, dim = 0) / time_steps).unsqueeze(0).repeat(time_steps,1,1)

    # Sparse, so initial guess is the zeros-tensor
    Y_0 = torch.zeros(time_steps, h, w)

    # Z_0 such that the linear constraints are satisfied
    Z_0 = video - X_0 - Y_0

    # Parameters
    beta = (10, 20, 50, 100)
    q = 0.001
    C_x = math.sqrt(h*w)
    # For l_{1}-norm, $lambda$
    lambd = 20
    # For Frobenius norm, $mu$
    mu = 0.5

    # Number of iterations
    niter = 100

    "RUN ALGORITHM FOR DIFFERENT BETA"

    # Array to store each value of beta
    tensor_admm = list()

    # Initialize instances of class TensorADMM for the different values of beta
    for i in range(len(beta)):
        tensor_admm.append(TensorADMM(video, X_0, Y_0, Z_0, param = (beta[i], q, C_x, lambd, mu)))

    # Execute the ADMM instances in different processes
    tensor_admm = Parallel(-1)(delayed(run)(instance, niter) for instance in tensor_admm)

    "PLOTTING OF LOGGED DATA"
    plot(tensor_admm, niter)

    # Summary statistics printed in console if needed
    # Could uncomment:
    summary_statistics(tensor_admm)

    "APPLICATION FOR BETA = 10 AFTER 50 AND 500 ITERATIONS"
    "If you want to run the detection for specific values of beta and for different numbers of iterations, "
    "please modify the values"
    
    beta = 10
    iterations = (50, 500)

    "To run, please uncomment next line"
    # detect(beta, iterations)

    
    

    
