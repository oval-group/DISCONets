import math
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np 
from chainer import cuda
import cupy

class Score(chainer.Chain):

	def __init__(self, beta = 0.5, alpha = 0.5, n_per_sample = 2):
		"""
		Arguments
		\beta : norm-parameter
		\alpha : dissimilarity coefficient parameter (\gamma in our paper)
		\n_per_sample : number of sampled outputs (K in our paper)
		"""
		super(Score, self).__init__()
		self.beta = beta
		self.alpha = alpha
		self.n_per_sample = n_per_sample

	def bnorm(self, Z):
		Z += 1e-20 #small constant for gradient stabilitly
		norm = F.basic_math.absolute(Z)
		norm = F.basic_math.pow(norm,2.0) # here we use 2.0 as the 2nd norm parameter, hence using a strictly proper scoring rule when \alpha = 0.5
		norm = F.sum(norm, axis = -1)
		norm = F.basic_math.pow(norm,self.beta/2.0)
		return norm

	def __call__(self, Ys, y):
		"""
		Arguments
		Ys -- (batchsize, n_per_sample, ...) sample outputs from the model.
		y -- (batchsize, ...) ground truth instances.
		"""
		n = y.data.shape[0]
		n_samples = Ys.data.shape[1]

		# create a 3D matrix
		y_mul = F.expand_dims(y, 1)
		y_mul_3d = F.concat((y_mul,), axis = 1)
		for k in range(n_samples - 1):
			y_mul_3d = F.concat((y_mul_3d, y_mul), axis = 1)
		Z_gt = Ys - y_mul_3d
		loss = F.sum(self.bnorm(Z_gt)) / n * (1.0 / n_samples)
		if self.alpha > 0.0:
			# create a 4D matrix
			Y_mul = F.expand_dims(Ys, 1)
			Y_mul_4d = F.concat((Y_mul,), axis = 1)
			for k in range(n_samples - 1):
				Y_mul_4d = F.concat((Y_mul_4d, Y_mul), axis = 1)
			Z = Y_mul_4d - F.swapaxes(Y_mul_4d,1,2)
			second_term = F.sum(self.bnorm(Z))
			second_term = - second_term / n * self.alpha * 1.0 / (n_samples * (n_samples - 1))
			loss = F.basic_math.add(loss,second_term)
		return loss

class WeightedScore(Score):

	def __init__(self, score, weighting):
		"""
		Arguments
		score : Score object to be weighted
		weighting : vector of weights, must be the same size as the output dimension
		"""
		super(WeightedScore, self).__init__()
		self.score = score
		self.weighting = chainer.Variable(weighting)
		self.beta = self.score.beta
		self.alpha = self.score.alpha
		self.n_per_sample = self.score.n_per_sample

	def __call__(self, Ys, y):
		"""
		Arguments
		Ys -- (batchsize, n_per_sample, ...) sample outputs from the model.
		y -- (batchsize, ...) ground truth instances.
		"""
		# weights the inputs data
		# the last dimension of the data must be the same as the weighting
		d_w = self.weighting.data.shape[0]
		d_y = y.data.shape[-1]
		b = y.data.shape[0]
		n_samp = Ys.data.shape[1]
		if not d_w == d_y:
			print("Weighting and dimension of output must be the same !")
		w = F.expand_dims(self.weighting, 0)
		w_2d  = F.broadcast_to(w,(b, d_w))
		y = F.basic_math.mul(y, w_2d)
		w_3d = F.expand_dims(w, 1)
		w_3d  = F.broadcast_to(w_3d,(b, n_samp, d_w))
		Ys = F.basic_math.mul(Ys, w_3d)
		return self.score(Ys,y)

def compute_score(score, Ys, y):
	# score is a Score object, created an attached to the model 
	return score(Ys, y)
