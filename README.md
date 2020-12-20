# Neural-network-for-topology-optimization

Deep learning based approach was proposed for accelerating the topology optimization problem. An encoder-decoder convolutional neural network was developed for this layout problem which performed the convergence of densities during topology optimization process.

A dual channel input image was used as an input for the model. First channel was the density distribution Xn which was the result of optimization process after n-iteration. Second channel was the gradient of the density (Xn - Xn-1) the difference between density distribution of last and its previous iteration. The output of the mdoel was the gray scale image with the same resolution as the input predicting the final optimized structure.

Results showed good accuracy and reduced the computational time for obtaining the final structure significantly.
