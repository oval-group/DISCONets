import os

if __name__ == '__main__':

	beta = 1.0
	seed = 0
	alpha = 0.5
	C = 1e-3
	savedir = './TryRelease'
	nrand = 200
	finger_w = 1.0
	fingers = ["Pinky,Ring,Middle,Palm,Index,Thumb"]

	cmd_str = "python hand_pose.py %f %f %s %f %d %d %s %f" % (beta, alpha, savedir, C, nrand, seed, fingers, finger_w) 
	os.system(cmd_str)
