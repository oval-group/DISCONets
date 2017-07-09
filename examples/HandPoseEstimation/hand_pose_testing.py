import numpy as np
import sys
sys.path.append('../utils/')
from utils import *
from data.dataset import NYUDataset
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import cupy
from hand_pose import JointPositionExtractor
import scipy

class DataForEvaluation(object):

	def __init__(self, x, y, testSeqs, indexes, sub_gt3D, J, name):

		self.x = x
		self.y = y
		self.testSeqs = testSeqs 
		self.indexes = indexes
		self.sub_gt3D = sub_gt3D
		self.J = J
		self.name = name

class GlobalEv(object):

	def __init__(self, datas, model, xp):

		self.x = datas.x
		self.y = datas.y
		self.model = model 
		self.xp = xp
		self.datas = datas

	def score_log_likelihood(self, bw = 'scott'):
		
		xp = self.xp
		n = self.y.shape[0]
		llh = np.zeros(n)
		for i in range(n):
			predictions_i = self.predicted_poses[i,:,:]
			predictions_i = cuda.to_cpu(predictions_i)
			dens_c_vector = contruct_parzen_window(predictions_i.T,bw)
			llh[i] = dens_c_vector.evaluate_log(self.y[i,:].T) 
		self.llh = llh
		return llh

	def get_point_estimates(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		self.poses_me = xp.zeros((n,d))
		self.poses_max = xp.zeros((n,d))
		self.poses_ff = xp.zeros((n,d))
		self.poses_scoring = xp.zeros((n,d))
		self.poses_sqrt_euc = xp.zeros((n,d))
		for i in range(n):

			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			y_me, y_max, y_ff, y_scoring, y_sqrt_euc = compute_MEU(self.predicted_poses[i,:,:], self.datas.J, self.score)
			self.poses_me[i,:] = self.predicted_poses[i,int(y_me),:]
			self.poses_max[i,:] = self.predicted_poses[i,int(y_max),:]
			self.poses_ff[i,:] = self.predicted_poses[i,int(y_ff),:]
			self.poses_scoring[i,:] = self.predicted_poses[i,int(y_scoring),:]
			self.poses_sqrt_euc[i,:] = self.predicted_poses[i,int(y_sqrt_euc),:]
	
	def compute_Losses_samples(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		N_sampling = self.predicted_poses.shape[1]
		probloss = np.zeros(n)
		sqrt_eu_err = np.zeros(n)
		for i in range(n):
			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			probloss[i] = compute_ProbLoss_alpha05(self.predicted_poses[i,:,:], y, self.score)
			sqrt_eu_err[i] = compute_SQRT_EuclidianError(self.predicted_poses[i,:,:], y)

		return probloss, sqrt_eu_err
	
	def compute_Losses_PE(self):

		n = self.y.shape[0]
		d = self.y.shape[1]
		N_sampling = self.predicted_poses.shape[1]
		probloss = np.zeros(n)
		sqrt_eu_err = np.zeros(n)
		for i in range(n):
			y = xp.asarray(self.y[i,:][xp.newaxis]).astype('float32')
			probloss[i] = compute_ProbLoss_alpha05(self.poses_scoring[i,:][np.newaxis].astype('float32'), y, self.score)
			sqrt_eu_err[i] = compute_SQRT_EuclidianError(self.poses_sqrt_euc[i,:][np.newaxis].astype('float32'), y)

		return probloss, sqrt_eu_err
	
	def get_point_estimates_3M(self, MEU = True, random = False):

		if not MEU:
			if not random:
				self.poses_me[:,:] = self.poses
				self.poses_max[:,:] = self.poses
				self.poses_ff[:,:] = self.poses
				self.poses_scoring[:,:] = self.poses
				self.poses_sqrt_euc[:,:] = self.poses
			else:
				n = self.predicted_poses.shape[0]
				n_predictions = self.predicted_poses.shape[1]
				for i in range(n):
					idx = np.random.randint(0,n_predictions)
					self.poses_me[i,:] = self.predicted_poses[i,idx]
					self.poses_max[i,:] = self.predicted_poses[i,idx]
					self.poses_ff[i,:] = self.predicted_poses[i,idx]
					self.poses_scoring[i,:] = self.predicted_poses[i,idx]
					self.poses_sqrt_euc[i,:] = self.predicted_poses[i,idx]
		else:
			self.get_point_estimates()

class Evaluator(GlobalEv):

	def __init__(self, datas, model, xp):

		self.x = datas.x
		self.y = datas.y
		self.model = model 
		self.xp = xp
		self.datas = datas
		self.score = model.score

	def generate_N_samples(self, N_sampling):

		xp = self.xp
		n = self.y.shape[0]
		d = self.y.shape[1]
		predictions = xp.zeros((n,N_sampling, d)).astype('float32')
		mean_predictions = xp.zeros((n, d)).astype('float32')
		seq = self.datas.testSeqs[0]
		for i in range(n):
			
			x_i = self.x[i,:,:,:][xp.newaxis]
			# Generate N_sampling poses for a depth image
			predictions_i = xp.asarray(xp.zeros((N_sampling, d))).astype('float32')
			x = chainer.Variable(xp.asarray(x_i).astype('float32')) 
			image_features = self.model.fast_sample_depth_image(x)
			image_features_all = chainer.Variable(xp.asarray(xp.tile(image_features.data, \
						(N_sampling,1)).astype('float32')))
			z_all = chainer.Variable(xp.asarray(np.random.uniform(-1, 1,(N_sampling, self.model.nrand)).astype(np.float32)))
			predictions_i = self.model.fast_sample(z_all, image_features_all)
			com_i = xp.asarray(seq.data[self.datas.indexes[i]].com)
			# reshape for denormalization
			predictions_i = xp.reshape(predictions_i.data,(N_sampling, self.datas.J, 3))*(seq.config['cube'][2]/2.) + com_i
			predictions_i = xp.reshape(predictions_i, ((N_sampling, self.datas.J * 3)))
			predictions[i,:,:] = predictions_i
			mean_predictions[i,:] = xp.mean(predictions_i, axis = 0)
		self.predictions = chainer.Variable(predictions.astype('float32'))
		self.predicted_poses = predictions
		self.poses = mean_predictions
		return predictions.data

class EvaluatorPointEstimateModels(GlobalEv):

	def __init__(self, datas, point_estimations, scoring, xp):

		self.x = datas.x
		self.y = datas.y
		self.xp = xp
		self.datas = datas
		self.poses = point_estimations
		self.score = scoring

	def denorm_point_estimations(self, normed_point_estimations):
		
		n = self.y.shape[0]
		d = self.y.shape[1]
		xp = self.xp
		point_estimations = xp.zeros((n, d)).astype('float32')
		seq = self.datas.testSeqs[0]

		for i in range(n):
			com_i = xp.asarray(seq.data[self.datas.indexes[i]].com)
			predictions_i = normed_point_estimations[i,:]
			# reshape for denormalization
			predictions_i = xp.reshape(predictions_i,(self.datas.J, 3))*(seq.config['cube'][2]/2.) + com_i
			predictions_i = xp.reshape(predictions_i, (self.datas.J * 3))
			point_estimations[i,:] = predictions_i

		return point_estimations

	def generate_N_samples(self, N_sampling, cov_value):

		# Generate N_sampling poses for a depth image
		n = self.y.shape[0]
		d = self.y.shape[1]
		xp = self.xp
		predictions = xp.zeros((n, N_sampling, d)).astype('float32')
		for i in range(n):
			x_i = self.x[i,:,:,:][xp.newaxis]
			y_i = self.poses[i,:][xp.newaxis]
			y_i_array = xp.asarray(xp.repeat(y_i, N_sampling, axis = 0))
			covariance = cov_value * np.eye(d).astype(np.float32)
			z = xp.asarray(scipy.stats.multivariate_normal.rvs(mean = np.zeros(d), cov = covariance, size = N_sampling))
			predictions[i,:,:] = y_i_array + z
		self.predictions = chainer.Variable(predictions)
		self.predicted_poses = predictions

def evaluate(results_dir, beta, seed, alpha, C, cov, MEU, random, results_dir, nrand, testSeqs, X_test, Y_test, N_sampling, J = 14):

	N = X_test.shape[0]
	distances = np.arange(0.0,81,5)
	line = np.zeros(len(distances))
	fraction_frame = 0.0
	probloss = np.zeros(N)
	sqrt_euc = np.zeros(N)
	probloss_PE = np.zeros(N)
	sqrt_euc_PE = np.zeros(N)
	euclidian_distance = np.zeros((N, J))
	max_euclidian_distance = np.zeros(N)
	
	gt3D = []
	for seq in testSeqs:
		gt3D.extend([j.gt3Dorig for j in seq.data])
	gt3D = np.asarray(gt3D)
	scoring = Score(beta, alpha, N_sampling) 
	model = JointPositionExtractor(scoring, nrand, J) 
	serializers.load_npz(results_dir + '/model_end.model', model)
						
	if use_gpu:
		model.to_gpu(gpu_id)

	with cupy.cuda.Device(gpu_id):
		start = 0
		end = 0
		batchsize = 1000
		while True:
			start = end
			end = min([start + batchsize, N])
			indexes = np.arange(start,end)
			sub_gt3D_array = xp.reshape(gt3D[indexes], (gt3D[indexes].shape[0], J * 3))
			sub_gt3D = gt3D[indexes]
			sub_gt3D.tolist()
			x_test = X_test[indexes,:,:,:]
			y_test = Y_test[indexes]
			datas = DataForEvaluation(x_test, sub_gt3D_array, testSeqs, indexes, sub_gt3D, J, dataset)
			if alpha > 0.0:
				evaluator = Evaluator(datas, model, xp)
				print("Generating samples")							
				evaluator.generate_N_samples(N_sampling) # Generate predictions, already denormed
			else:
				evaluator_PE = Evaluator(datas, model, xp)
				evaluator_PE.generate_N_samples(1) # get PE
				point_estimations = xp.squeeze(evaluator_PE.predicted_poses, axis = 1)
				evaluator = EvaluatorPointEstimateModels(datas, point_estimations, scoring, xp)	
				print("Generating samples")
				evaluator.generate_N_samples(N_sampling, cov_value = cov)

			print("Computing Probabilistic metrics")	
			l1, l2 = evaluator.compute_Losses_samples() 
			probloss[start:end] = l1 
			sqrt_euc[start:end] = l2 

			print("Computing NON Probabilistic metrics")	
			evaluator.get_point_estimates_3M(MEU, random) # generate point estimates
			l1, l2 = evaluator.compute_Losses_PE() 
			probloss_PE[start:end] = l1 
			sqrt_euc_PE[start:end] = l2 
				
			print("Computing Errors")	
			m1,m2,m3 = values_in_mm(evaluator)
			m1 = cuda.to_cpu(m1)
			m2 = cuda.to_cpu(m2)
			euclidian_distance[start:end, :] = m1
			max_euclidian_distance[start:end] = m2 
			fraction_frame += m3 / float(N) 
			
			print("Computing FF curves")	
			line_bi = compute_ff_line_PE(evaluator_PE.predicted_poses, sub_gt3D_array, 14, distances)
			
			line += line_bi / float(N)
			
			if end == N:
				break

def l2norm(Z, beta = 1.0): # the loss can be different than the one used during training 
	
	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0)
	return norm

def euclidian_joints(Z):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) 
	return norm.data

def max_euclidian_joints(Z):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) 
	norm = F.max(norm, axis =-1) 
	return norm.data

def number_frames_within_dist(Z, dist = 80):

	norm = F.basic_math.absolute(Z)
	norm = F.basic_math.pow(norm,2.0)
	norm = F.sum(norm, axis=-1)
	norm = F.basic_math.pow(norm,1.0/2.0) # shape[-1] is J
	norm = F.max(norm, axis =-1) 

	n_frames = xp.sum((norm.data <= dist), axis = -1)

	return n_frames

def compute_ff_line_PE(predictions, groundtruth, J, distances):

	N = predictions.shape[0]
	n_predictions = predictions.shape[1]
	assert n_predictions == 1
	d = predictions.shape[2]
	line = np.zeros(len(distances))
	for i in range(N):

		prediction_i = predictions[i,:,:]	
		gt = groundtruth[i,:]
		for idx_dist, dist in enumerate(distances):
			y_ff = 0
			pe = prediction_i[int(y_ff), :]
			delta = chainer.Variable(xp.asarray(gt) - pe)
			delta = F.reshape(delta, (delta.data.shape[:-1] + (J, 3)))
			max_dist = max_euclidian_joints(delta) 
			line[idx_dist]+= (max_dist<=dist)

	return line

def compute_ff_line(predictions, groundtruth, J, distances):

	N = predictions.shape[0]
	n_predictions = predictions.shape[1]
	d = predictions.shape[2]
	line = np.zeros(len(distances))
	for i in range(N):

		prediction_i = predictions[i,:,:]	
		gt = groundtruth[i,:]
		predictions_3d_matrix = xp.repeat(prediction_i,n_predictions,axis=0).astype(np.float32)
		predictions_3d_matrix = predictions_3d_matrix.reshape(n_predictions,n_predictions,d)
		Z = predictions_3d_matrix - prediction_i
		Z_flatten = chainer.Variable(Z)
		Z = F.reshape(Z_flatten, (Z_flatten.data.shape[:-1] + (J, 3)))
		# Compute the max on the L2 norm
		norm = max_euclidian_joints(Z) 
		# norm is of size 100, 100
		for idx_dist, dist in enumerate(distances):
			fraction_frame_loss_matrix = xp.sum((norm <= dist), axis = -1) 
			y_ff = xp.argmax(fraction_frame_loss_matrix)
			pe = prediction_i[int(y_ff), :]
			delta = chainer.Variable(xp.asarray(gt) - pe)
			delta = F.reshape(delta, (delta.data.shape[:-1] + (J, 3)))
			max_dist = max_euclidian_joints(delta) 
			line[idx_dist]+= (max_dist<=dist)

	return line

def compute_MEU(predictions, J, score):

	# create matrix of losses
	n_predictions = predictions.shape[0]
	d = predictions.shape[1]
	
	predictions_3d_matrix = xp.repeat(predictions,n_predictions,axis=0).astype(np.float32)
	predictions_3d_matrix = predictions_3d_matrix.reshape(n_predictions,n_predictions,d)
	
	Z = predictions_3d_matrix - predictions
	Z_flatten = chainer.Variable(Z)
	Z = F.reshape(Z_flatten, (Z_flatten.data.shape[:-1] + (J, 3)))
	
	mean_per_joint_error_loss_matrix = xp.sum(euclidian_joints(Z), axis = -1) / J
	mean_error_loss_matrix = euclidian_joints(Z_flatten) / (3*J)
	max_error_loss_matrix = max_euclidian_joints(Z)
	training_scoring_matrix = score.bnorm(Z_flatten).data
	fraction_frame_loss_matrix = number_frames_within_dist(Z)

	MEU_vector_scoring = xp.sum(training_scoring_matrix, axis = 1)
	MEU_vector_me_per_joint = xp.sum(mean_per_joint_error_loss_matrix, axis = 1)
	MEU_vector_max = xp.sum(max_error_loss_matrix, axis = 1)
	MEU_vector_me = xp.sum(mean_error_loss_matrix, axis = 1)

	y_me_per_joint = xp.argmin(MEU_vector_me_per_joint)
	y_me = xp.argmin(MEU_vector_me)
	y_max = xp.argmin(MEU_vector_max)
	y_ff = xp.argmax(fraction_frame_loss_matrix)
	y_scoring = xp.argmin(MEU_vector_scoring)

	return y_me_per_joint, y_max, y_ff, y_scoring, y_me

def compute_SQRT_EuclidianError(predictions, gt):

	n_predictions = predictions.shape[0]
	Zgt = predictions - gt
	Zgt = chainer.Variable(Zgt)
	term1 = F.sum(l2norm(Zgt, 1.0)) / n_predictions
	return term1.data

def compute_ProbLoss_alpha05(predictions, gt, score):

	n_predictions = predictions.shape[0]

	if n_predictions == 1:
		term2 = chainer.Variable(cuda.cupy.asarray(0.0).astype('float32'))
	else:
		d = predictions.shape[1]
		predictions_3d_matrix = xp.repeat(predictions,n_predictions,axis=0).astype(np.float32)
		predictions_3d_matrix = predictions_3d_matrix.reshape(n_predictions,n_predictions,d)
		Z = predictions_3d_matrix - predictions
		Z = chainer.Variable(Z)
		term2 = F.sum(score.bnorm(Z)) / (n_predictions * (n_predictions - 1))
	
	Zgt = predictions - gt
	Zgt = chainer.Variable(Zgt)
	term1 = F.sum(score.bnorm(Zgt)) / n_predictions
	obj = term1 - 0.5 * term2 
	return obj.data

def values_in_mm(evaluator):
	
	joints_me = xp.reshape(evaluator.poses_me, (evaluator.poses_me.shape[0], evaluator.datas.J,3))
	joints_max = xp.reshape(evaluator.poses_max, (evaluator.poses_max.shape[0], evaluator.datas.J,3))
	joints_ff = xp.reshape(evaluator.poses_ff, (evaluator.poses_ff.shape[0], evaluator.datas.J,3))
	
	gt = xp.asarray(evaluator.datas.sub_gt3D).astype('float32')
	vector_me = chainer.Variable((joints_me - gt).astype('float32'))
	vector_max = chainer.Variable((joints_max - gt).astype('float32'))
	vector_ff = chainer.Variable((joints_ff - gt).astype('float32'))
	
	euclidian_distance = euclidian_joints(vector_me)
	max_error = max_euclidian_joints(vector_max)
	ff_error = number_frames_within_dist(vector_ff)

	return euclidian_distance, max_error, ff_error

if __name__ == '__main__':

	global gpu_id
	dataset = 'test'
	datadir = '.'
	di = NYUImporter(datadir)
	Seq = di.loadSequence(dataset)
	testSeqs = [Seq]
	testDataset = NYUDataset(testSeqs)
	use_gpu = True
	gpu_id = 0
	if use_gpu:
		xp = cuda.cupy
	else:
		xp = np
	J = 14
	X_test, Y_test = testDataset.imgStackDepthOnly(dataset)
	Y_test = xp.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]* Y_test.shape[2]))
	np.random.seed(0)
	beta = 1.0
	seed = 0
	alpha = 0.5
	C = 1e-3
	nrand = 200
	evaluate(results_dir, beta, seed, alpha, C, cov, MEU, random, results_dir, nrand, testSeqs, X_test, Y_test, N_sampling, J = 14)