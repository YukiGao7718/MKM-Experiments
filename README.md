# MKM-Experiments
It is contributed by Siyu Gao (sg5578@nyu.edu) at New York University Shanghai.  

This is data science project focuses on testing multilayer kernel machines (MKM) following the results in [Kernel Methods in Deep Learning](https://proceedings.neurips.cc/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf).  

Abstract
-----------------
MKM is proposed as a combined algorithm of Kernel PCA and Feature Selection with a final Classifier. We not only reproduce the MKM following the way in original paper, but also change the design of MKMs and tested these new designs on different kernels and classifiers. Experiments are carried mainly on MNIST dataset for classification problem. It turns out that the original MKM algorithm is very sensitive to hyper-parameter of feature selection which can significantly undermine the accuracy. For general MKMs, linear and polynomial kernels performs quite well. For MKMs with Arc-cosine Kerenl, lower order with less layer always outperform others,which is consistent with results in original paper. Also, we found that model KernelPCA+KNN with arc-cosine kernel at order 0 with only 1 layer achieves the best stable performance.  
More details can be found in file `Final report.pdf`. 

Presenations
-----------------
* `presentation1.pdf` is mainly a summary of [Kernel Methods in Deep Learning](https://proceedings.neurips.cc/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf). 
* `presentation2.pdf` represents the underlying theory and realization of deep kernel machines.

Acknowledgement
-----------------

This project is finished during Spring 2021 Math Foundations for Machine Learning and Data Science course. I appreciate Prof.Shuayang Ling a lot for all the instructions and advice he provided to this project.
