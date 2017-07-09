This repository contains the code associated to the DISCO Nets paper that can be found on arxiv.

# Requirements:
 
* Python + classic packages (numpy)

* [Chainer](http://chainer.org/) library for Python

If you use this work, please cite:

D. Bouchacourt, M. P. Kumar, S. Nowozin, "DISCO Nets: DISsimilarity COefficient Networks", NIPS 2016

# Repository contents

* utils/scores.py : define here additional scoring function if needed. We have implemented the \alpha -\beta norm with \alpha = 2 (as used in our experiment)
* examples/HandPoseEstimation/train.py : launching script
* examples/HandPoseEstimation/hand_pose.py : defines the DISCO Nets and runs training
* examples/HandPoseEstimation/hand_pose_testing.py : testing utils specific to hand pose estimation

# Run a simple example :

The "example" folder allows you to use the DISCO Nets on the NYU Hand Pose dataset [NYU Hand Pose dataset](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm). Data should be pre-processed using the code from [Markus Oberweger](https://cvarlab.icg.tugraz.at/projects/hand_detection/) and require the installation of DeepPrior to load the data.

If you want to use a GPU, set up **gpu = True** and your gpu ID in the file **hand_pose.py**. To run the example, set up your parameters in the file **train.py** and run **python train.py** from your terminal. 

Parameters:

* 	beta : same as in the paper.
*	seed : random seed to initialise the network weights. All biases are initialised to 0.
*	alpha : dissimilarity coefficient hyper-parameter, referred as gamma in the paper.
*	C : weight decay 
*	savedir : folder to save the model + monitored values at each iteration
*	datadir : folder to find the data, in the NYU Hand Pose example "../"
*	nrand : size of the noise vector
*	finger_w : used in our fingers experiment, leave it to 1.0 if you want to consider all 5 fingers
*	fingers :  used in our fingers experiment, leave it to the full list of fingers if you want to consider all 5 fingers