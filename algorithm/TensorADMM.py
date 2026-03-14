# Tensor processing
import torch

# For elementary operations
import math
import numpy as np

# For 1D-solver of the X-subproblem
import scipy

# For measuring time, if needed
import time

"""
Class for Optimizer, handles mathematical part of the object detection application and stores summary statistics in form of graphs
    - Takes initial guesses as tensors
    - Stores primal and dual variables 
    - Implements the subproblems
    - Logs needed quantities
"""

class TensorADMM:

    def __init__(self, video, X_initial, Y_initial, Z_initial, param):
        self.video = video

        # Number of frames, height, width
        self.time_steps, self.M, self.N = video.shape

        # Initial guesses
        self.X_, self.Y_, self.Z_ = X_initial, Y_initial, Z_initial

        # Constants from algorithm: beta, q, C_{X}, lambda, mu
        self.beta, self.q, self.C_x, self.lambd, self.mu = param

        # Lipschitz Constant From Lemma 5.2
        # C = (smallest eigenvalues of I) * L_h * (sm. eigenvalue of I) = 1 * L_h * 1 = 2*mu
        self.C = 2*self.mu

        # Lemma 5.1 inverse Lipschitz constant
        self.M_bar = 1 # Identity mapping as linear operators

        # Inital Condition
        self.W_ = -2*self.mu*self.Z_

        # Logging
        # Dictionary for each quantity for saving intermediate values of augmented Lagrangian and linear equality constraint violation
        # and other quantities of interest
        self.logging = {
        "values_aug_lagr": [], # augmented Lagrangian values

        "values_descent": [],
        "values_descent_lower": [], # lower bound on descent from proof

        "values_subgr":  [],
        "values_subgr_upper" : [], # upper bound on subgradient norm

        # Measures feasibility violation
        "values_feas":  [],

        # Objective
        "objective": [],

        # Measures the relation in Lemma 5.2.1 
        "values_primal_dual" : []
        }

        # Log initial values
        self.log_first()
    
    """
    OBJECTIVE / AUGMENTED LAGRANGIAN EVALUATION
    """

    # X-Subproblem

    # Schatten q-norm of a matrix (slice) X_fft_j in Fourier domain
    def schattennorm_slice(self, X_fft_j):
        singular_values = np.linalg.svdvals(X_fft_j)
        singular_values_q_power = np.power(singular_values, self.q)
        return self.C_x*np.sum(singular_values_q_power)
    
    # Tensorial Schatten q-norm
    # to the q-th power
    def schattennorm(self, X):
        # FFT of X along time slices / 0th dimension
        X_fft = torch.fft.fftn(X)

        # Stores value
        sum_of_slices = 0

        # Add values of individual slices in Fourier domain
        for l in range(self.time_steps):
            sum_of_slices += self.schattennorm_slice(X_fft[l,:,:])

        # Normalize by T, see definition of tensorial Schatten-q-norm
        return sum_of_slices / self.time_steps
    
    # Objective
    def objective(self):
        obj_x = self.schattennorm(self.X_)
        obj_y = self.lambd * torch.linalg.vector_norm(self.Y_, ord = 1)
        obj_z = self.mu * torch.linalg.vector_norm(self.Z_, ord = 2)**2

        return  obj_x + obj_y + obj_z

    # Augmented Lagrangian
    def augmented_lagrangian(self):
        # Constraint Violation
        R = self.X_ + self.Y_ + self.Z_ - self.video
        
        # Inner product term with dual variable w
        dual = torch.sum(self.W_ * R).item()

        # Augmented penalty
        augmented = self.beta / 2 * torch.linalg.vector_norm(R)**2

        # Objective
        obj = self.objective()
        
        return obj + dual + augmented 
    
    """
    ALGORITHM + LOGGING
    """
    # One iteration
    def step(self):
        # Store augmented Lagrangian before this iteration
        prev_aug_lagr = self.augmented_lagrangian()

        # Copy prev. iterates before this iteration, only from running the algorithm will they be instantiated
        # Needed to log residual norms

        self.X_prev = self.X_.clone().detach()
        self.Y_prev = self.Y_.clone().detach()
        self.Z_prev = self.Z_.clone().detach()

        # Solve subproblems:

        # Primal Updates
        self.X_ = self.X_Subproblem()
        # print(f"X: {self.augmented_lagrangian()}")
        self.Y_ = self.Y_subproblem()
        # print(f"Y: {self.augmented_lagrangian()}")
        self.Z_ = self.Z_subproblem()
        # print(f"Z: {self.augmented_lagrangian()}")

        # Dual Update
        self.W_ = self.W_ + self.beta*(self.X_ + self.Y_ + self.Z_ - self.video)

        # return previous augmented Lagrangian value to plot descent
        return prev_aug_lagr
    
    # Logging methods
    def log_first(self):
        lag_current = self.augmented_lagrangian()

        self.logging["values_aug_lagr"].append(lag_current)
        self.logging["objective"].append(self.objective())

        # Constraint violation
        R_helper = self.video - self.X_ - self.Y_ - self.Z_

        self.logging["values_feas"].append(torch.linalg.vector_norm(R_helper))

        # Primal-Dual Relation
        R_helper = self.W_ + 2*self.mu*self.Z_
        self.logging["values_primal_dual"].append(torch.linalg.vector_norm(R_helper))
   
        return lag_current
    
    def log_data(self, lag_prev):
        # lag_prev: augm Lagrangian value of previous iterates

        lag_current = self.augmented_lagrangian()
        self.logging["values_aug_lagr"].append(lag_current)
        self.logging["objective"].append(self.objective())

        self.logging["values_descent"].append(lag_prev - lag_current)
        self.logging["values_descent_lower"].append(self.residual_norm_squared() / (self.M_bar)**2) # Due to proximal ADMM

        # Subgradient Norm

        # Componentwise Subgradients by Proposition 2
        x_subgr_aug = self.subgradients_x() 
        y_subgr_aug = self.subgradients_y()
        z_subgr_aug = self.subgradients_z()
        w_subgr_aug = self.subgradients_w()

        subgr_norm = math.sqrt(torch.linalg.vector_norm(x_subgr_aug)**2 + torch.linalg.vector_norm(y_subgr_aug)**2 + torch.linalg.vector_norm(z_subgr_aug)**2
                               + torch.linalg.vector_norm(z_subgr_aug)**2)
        self.logging["values_subgr"].append(subgr_norm)
        
        # Upper Bound
        
        self.logging["values_subgr_upper"].append(self.subgr_constant()*self.residual_norm()) 
        
        # Feasibility violation
        R_feasibility = self.X_ + self.Y_ + self.Z_ - self.video
        self.logging["values_feas"].append(torch.linalg.vector_norm(R_feasibility))
        
        # Primal-Dual violation, cf. Lemma 5.2
        R_primal_dual = self.W_ + 2*self.mu*self.Z_
        self.logging["values_primal_dual"].append(torch.linalg.vector_norm(R_primal_dual))

        return lag_current
        
    # Full run
    def algorithm(self, niter):
        for i in range(niter):

            prev_aug_lagr = self.step()

            # Logging
            prev_aug_lagr = self.log_data(prev_aug_lagr) 
        return prev_aug_lagr

    def algorithm_and_measurements(self, niter):
        # Measure time
        start_time = time.time()
        startval = self.logging["values_aug_lagr"][0]

        # Algorithm
        endval = self.algorithm(niter)

        # Time measure
        end_time = time.time()

        # Needed time
        print(f"It took {end_time - start_time} seconds, \n starting with value {startval} and ending with {endval}.")

        return 0
    
        # Logging helpers
    def residual_norm_squared(self):
        X_res = self.X_ - self.X_prev
        Y_res = self.Y_ - self.Y_prev
        Z_res = self.Z_ - self.Z_prev

        return torch.linalg.vector_norm(X_res)**2 + torch.linalg.vector_norm(Y_res)**2 + torch.linalg.vector_norm(Z_res)**2
    
    def residual_norm(self):
        X_res = self.X_ - self.X_prev
        Y_res = self.Y_ - self.Y_prev
        Z_res = self.Z_ - self.Z_prev

        return torch.linalg.vector_norm(X_res) + torch.linalg.vector_norm(Y_res) + torch.linalg.vector_norm(Z_res)
    

    """
    SUBPROBLEM SCHEMES
    """

    # X-Subproblem

    # Proximal cost function of the 1D-problem as seen in Chapter 6
    def q_objective(self, t, *args):
        # Point where proximal operator should be evaluated
        s = args[0]

        # Parameter for proximal term
        c = args[1]

        # Weights for Schatten-q-norm
        w = args[2]

        return w*abs(t)**self.q + c / 2 * (t - s)**2

    # Sign function for our purpose
    def sign(self, t):
        if t==0:
            return 0
        elif t < 0:
            return -1
        else:
            return 1

    # First Derivative for t > 0; for t = 0 return 0

    def q_derivative(self, t, *args):
        # Point where proximal operator should be evaluated
        s = args[0]

        # Parameter for proximal term
        c = args[1]

        # Weights for Schatten-q-norm
        w = args[2]

        if t == 0:
            return 0
        else:
            return w* self.q * self.sign(t) / (abs(t)**(1-self.q)) + c*(t - s)
        
    def q_hessian(self, t, *args):
        # Point where proximal operator should be evaluated
        s = args[0]

        # Parameter for proximal term
        c = args[1]

        # Weights for Schatten-q-norm
        w = args[2]

        if t == 0:
            return c
        else:
            return w * self.q*(self.q-1) / abs(t)**(2-self.q) + c
        
    # Solve onedimensional proximal problem at s >= 0 with penalty parameter c, weight w to schatten-objective
    def q_prox(self,t0, s, c, w):
        # Solve subproblem
        sol = 0
        # Prox(0) = 0 as seen in thesis
        threshold = (s != 0)
        # Use unconstrained Newton-CG, as it uses the exact hessian
        if threshold:
            sol = scipy.optimize.minimize(self.q_objective, x0 = t0, args = (s, c, w), method='Newton-CG', 
                                      jac = self.q_derivative, hess=self.q_hessian).x
        
        return abs(sol[0])

    def X_Slice_proximal(self, Slice_fft_t):
        # SVD of the t-th time slice in Fourier domain
        U, S, Vh = torch.linalg.svd(Slice_fft_t)

        # Vector of diagonal entries of Sigma_f from the construction in chapter 6
        S_f_t = torch.zeros(len(S), dtype = torch.cfloat)
        intermediate = 0 # helper to decouple prox-solution and string into vector S_f_t

        # For each singular value in S -> paralleizable (TO-DO)
        for i in range(len(S)):
            # t0
            s = S[i].item().real
            t0 = s
            # Complex entry with zero imaginary part to apply inverse FFT, then gets real by Kilmer
            intermediate = self.q_prox(t0, s, 2*self.beta, w = self.C_x) + 0j
            S_f_t[i] = intermediate

        Optimal_x = (U[:,:len(S)] * S_f_t) @ Vh

        return Optimal_x
    
    def X_Tensor_proximal(self, A):
        # Fiberwise FFT along time dimension (=0)
        A_fft = torch.fft.fftn(A, dim = 0)
        A_copy = A_fft.clone().detach().type(torch.cfloat)

        # Sequential
        for t in range(self.time_steps):
           A_fft[t, :, :] = self.X_Slice_proximal(A_copy[t, :, :])
        
        # Parallel
        #Result_Subscheme = Parallel(-1)(delayed(self.X_Slice_proximal)(A_copy[t,:,:]) for t in range(self.time_steps)) # Future
        #A_fft = torch.stack(Result_Subscheme, dim = 0)

        # Return real part of inverse FFT (imaginary part will be zero by Kilmer et al.)
        return torch.fft.ifftn(A_fft, dim = 0).real
    
    def X_Subproblem(self):
        # See algorithm
        A = ((self.video - self.Y_ - self.Z_ - self.W_ / self.beta) + self.X_) / 2
        return self.X_Tensor_proximal(A)
    
    # Y-Subproblem
    def soft_thresholding(self, lambd_bar, A):
        # See formula in Thesis in subsection "Y-subproblem"
        Optimal_y = torch.sign(A)*torch.maximum(torch.sub(A.abs(), lambd_bar), torch.tensor(0))
        return Optimal_y
    
    def Y_subproblem(self):
        A = ((self.video - self.X_ - self.Z_ - self.W_ / self.beta) + self.Y_)/2
        return self.soft_thresholding(self.lambd / (2*self.beta), A)
    
    # Z-Subproblem
    def Z_subproblem(self):
        A = self.beta*(self.video - self.X_ - self.Y_) - self.W_ 
        return A/ (2*self.mu + self.beta)
    

    """
    Subgradients of the subproblems
    """
    
    def subgr_constant(self):
        # See Proof of Lemma 5.8
        cons_w = self.C / self.beta # since zeta_y = 0 for exact solutions, ||B||=1
        cons_y = 2*self.mu # L_h, since zeta_y = 0

        # Constant from x-components
        cons_x = max(2*self.mu + self.beta, 
                 self.beta + 0 + self.beta/2) # C+beta or beta + L_g + ||H_i||

        # Triangle inequality, 4 components X,Y,Z,W for upper bound
        return 4*max(cons_w, cons_y, cons_x)

    
    # Subgradient of proximal X-subproblem
    def subgradients_x(self):
        "Derivation"
        # At time end of iteration k+1 -> prev corresponds to iteration k
        # X/x solves previous subproblem, hence: 0 \in \partial_{X} ||x||_{q}^{q} + 2*beta*x + W_prev + beta*(Y_prev + Z_prev - video - X_prev)
        # So beta*(X_prev + video - Y_prev - Z_prev) - W_prev - 2*beta*x \in \partial_{X} ||x||_{q}^{q} =: Subgradient at minimization problem

        # So subgradient at k+1 of L_beta: \partial_{X} ||x||_{q}^{q} + W_ + beta(x + Y_ + Z_ - video)
        # = beta*(x + X_prev + Y_ - Y_prev + Z_ - Z_prev)- 2*beta*x + W_ - W_prev 
        # = beta*(X_prev - x + Y_ - Y_prev + Z_ - Z_prev) + W_ - W_prev
        # = beta*(X_prev - x + Y_ - Y_prev + Z_ - Z_prev + x + Y_ + Z_ - video)
        # = beta*(X_prev + 2*Y_ - Y_prev + 2*Z_ - Z_prev - video)

        return self.beta*(self.X_prev + 2*self.Y_ - self.Y_prev + 2*self.Z_ - self.Z_prev - self.video)

    # Subgradient of proximal Y-subproblem, based on Proof of Lemma 5.
    def subgradients_y(self):
        "Derivation"
        # Y/y solves previous subproblem, i.e. 0 \in \partial_y L_{beta}(X_,y,Z_prev,W_prev) = lamb * \partial_y ||.||_{1}(y) + 2*beta*y + W_prev + beta*(X_ + Z_prev - video - Y_prev)
        # Thus - W_prev + beta*(video - X_  - Z_prev + Y_prev) - 2*beta*y \in lamb * \partial_y ||.||_{1}
        # prev_sub_mul_lamb = beta*(video - X_ - Z_prev + Y_prev) - 2*beta*y - W_prev
        #print(prev_sub_mul_lamb)

        "Proceeding:"
        # return prev_sub_mul_lamb + beta*(X_ + Y + Z_ - video) + W_
        # return beta*(Y + Y_prev + Z_ - Z_prev) - 2*beta*y + (W_ - W_prev)
        # y = Y, our notation
        # return beta*(Y_prev - y + Z_ - Z_prev) + (W_ - W_prev)
        # return beta*( Y_prev - y  + Z_ - Z_prev + X_ + y+ Z_ - video)
        # return beta*( Y_prev + 2*Z_ - Z_prev + X_ - video)

        return self.beta*(self.Y_prev + 2*self.Z_ - self.Z_prev + self.X_ - self.video)

    def subgradients_z(self):
        return (2*self.mu+ self.beta)*self.Z_ + self.beta*(self.X_ + self.Y_ - self.video) + self.W_

    def subgradients_w(self):
        return self.X_ + self.Y_ + self.Z_ - self.video 
