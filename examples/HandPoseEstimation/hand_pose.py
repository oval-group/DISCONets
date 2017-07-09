import numpy as np
import time
import sys
###################### ADD PATH TO THE DEEP PRIOR PACKAGE HERE
sys.path.append('../../../DeepPrior/src/')
sys.path.append('../../utils/')
from data.dataset import NYUDataset
from data.importers import NYUImporter
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer import serializers
import cupy
from scores import *

n_conv = 5
n_filters = 8
n_pool = 3
n_pixels_1 = 128
n_pixels_2 = 128
size_out_1 = 9
size_out_2 = 9

class JointPositionExtractor(chainer.Chain):
	
	def __init__(self, score, nrand, J = 14):
		super(JointPositionExtractor, self).__init__(
			# image network
			conv0 = L.Convolution2D(1, n_filters, n_conv, stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv)),
			conv1 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			conv2 = L.Convolution2D(n_filters,n_filters,n_conv,stride = 1, pad=0, wscale=0.01*math.sqrt(n_conv*n_conv*n_filters)),
			
			# start concatenating
			lin2 = L.Linear(size_out_1*size_out_2*n_filters + nrand, 1024, wscale=0.01*math.sqrt(size_out_1*size_out_2*n_filters + nrand)),
			lin3 = L.Linear(1024, 1024, wscale=0.01*math.sqrt(1024)),
			lin4 = L.Linear(1024, 3*J, wscale=0.01*math.sqrt(1024)),

		)
		self.es = None
		self.J = J
		self.nrand = nrand
		self.score = score
		self.n_per_sample = self.score.n_per_sample

	def fast_sample_depth_image(self, x, test=False):

		h_image = F.max_pooling_2d(F.relu(self.conv0(x)), ksize = n_pool)
		h_image = F.max_pooling_2d(F.relu(self.conv1(h_image)), ksize = n_pool)
		h_image = F.relu(self.conv2(h_image))
		h_image = F.reshape(h_image, (h_image.data.shape[0], size_out_1*size_out_2*n_filters))
		return h_image

	def fast_sample(self, z, image_features, test=False):
		h = F.concat((z,image_features), axis = 1)
		h = F.relu(self.lin2(h))
		h = F.relu(self.lin3(h))
		h = self.lin4(h)
		return h

	def __call__(self, x, y, test=False):

		features = self.fast_sample_depth_image(x)
		z = chainer.Variable(xp.random.uniform(-1.0, 1.0,
		(x.data.shape[0], self.nrand)).astype(xp.float32))
		Y = self.fast_sample(z, features)
		Y = F.expand_dims(Y, 1)
		Y_mul = F.concat((Y,))
		
		for k in range(self.n_per_sample - 1):
			z = chainer.Variable(xp.random.uniform(-1.0, 1.0,
			(x.data.shape[0], self.nrand)).astype(xp.float32))
			Y = self.fast_sample(z, features)
			Y = F.expand_dims(Y, 1)
			Y_mul = F.concat((Y_mul, Y), axis = 1)

		es = compute_score(self.score, Y_mul, y) # call scoring function
		self.es = es.data
		return es

def weight_on_fingers(fingers, weight):
	
	joints = ["Pinky tip", "Pinky mid", "Ring tip", "Ring mid", "Middle tip", "Middle mid", "Index tip", "Index mid", \
		"Thumb tip", "Thumb mid", "Thumb root", "Palm left", "Palm right", "Palm"]
	J = len(joints)
	weighting =  weight * xp.ones((J,3)).astype(xp.float32)
	for j in range(J):
		finger_name = joints[j].split(" ")[0]
		if finger_name in fingers:
			weighting[j,:] = 1.
	weighting = xp.reshape(weighting, 3*J)
	return weighting

def objective_function(model, x_val, y_val, n_per_sample_val, z_monitor_objective_all_val):
	
	N_val = x_val.shape[0]
	start = 0
	end = 0
	batchsize = 1000
	loss = 0.0
	while True:
		start = end
		end = min([start+batchsize,N_val])
		x = chainer.Variable(x_val[start:end,:,:,:].astype(xp.float32))
		y = chainer.Variable(y_val[start:end,:])
		ex = model.fast_sample_depth_image(x)
		z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, 0, :]))
		pred = model.fast_sample(z,ex)
		pred = F.expand_dims(pred, axis = 1)
		preds = F.concat((pred,))
		for k in range(n_per_sample_val - 1):
			z = chainer.Variable(xp.asarray(z_monitor_objective_all_val[start:end, k+1, :]))
			pred = model.fast_sample(z,ex)
			pred = F.expand_dims(pred, axis = 1)
			preds = F.concat((preds,pred), axis = 1)
		loss += model.score(preds,y).data * (end - start)
		if end == N_val:
			break
	return loss / N_val
	
if __name__ == '__main__':

	beta = float(sys.argv[1])
	alpha = float(sys.argv[2])
	savedir = sys.argv[3]
	C = float(sys.argv[4])
	nrand = int(sys.argv[5])
	seed = int(sys.argv[6])
	fingers = sys.argv[7].split(",")
	weight = float(sys.argv[8])

	n_per_sample = 2
	if alpha > 0.0:
		n_per_sample_val = 2
	else:
		n_per_sample_val = 1
	
	tol = 1e-12
	J = 14

	di = NYUImporter('../../../DeepPrior/')
	Seq = di.loadSequence('train')
	trainDataset = NYUDataset([Seq])
	X_train, Y_train = trainDataset.imgStackDepthOnly('train')

	use_gpu = True
	gpu_id = 0
	if use_gpu:
		xp = cuda.cupy
	else:
		xp = np
	
	Y_train = xp.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]* Y_train.shape[2]))
	
	np.random.seed(0)
	# shuffle the training samples
	indexes = np.arange(X_train.shape[0])
	np.random.shuffle(indexes)
	Nval = 10000 # save 10000 for validation set
	indexes_val = indexes[:Nval]
	indexes_train = indexes[Nval:]
	x_val = xp.asarray(X_train[indexes_val,:,:,:]).astype(xp.float32)
	y_val = xp.asarray(Y_train[indexes_val,:]).astype(xp.float32)
	x_train = X_train[indexes_train,:,:,:]
	y_train = Y_train[indexes_train,:]

	N = x_train.shape[0]
	#Create random noise to evaluate current objective function
	z_monitor_all_val = xp.asarray(np.random.uniform(-1.0, 1.0,
						(Nval, n_per_sample_val, nrand)).astype(np.float32))

	np.random.seed(seed)
	# score function 
	scoring = Score(beta, alpha, n_per_sample)
	weighting = weight_on_fingers(fingers, weight)	
	if weight != 1.0: 
		# using a weighted scoring slows a little the training, so use it only if there is a different weighting
		weighted_scoring = WeightedScore(scoring,weighting)
		model = JointPositionExtractor(weighted_scoring, nrand, J)
	else:
		model = JointPositionExtractor(scoring, nrand, J)
	if use_gpu:
		model.to_gpu(gpu_id)
	opt_model = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	
	opt_model.setup(model)
	opt_model.add_hook(chainer.optimizer.WeightDecay(C))
	
	batchsize = 256
	size_epoch = int(N/batchsize) + 1
	monitor_frequency = 10
	Nepoch = 400
	Probloss_val = xp.asarray(np.zeros(Nepoch))
	Probloss_train = xp.asarray(np.zeros(Nepoch))
	start_at = time.time()
 	print "Starting training..."
	with cupy.cuda.Device(gpu_id):
		epoch = 0
		period_start_at = time.time()
		bi = 0
		curr_epoch = 0
		while True:
			#monitor objective value
			if bi % size_epoch == 0:
				if curr_epoch % monitor_frequency == 0 or curr_epoch == (Nepoch-1):
					serializers.save_npz(savedir + '/model_%d.model' % curr_epoch, model) # save model every epoch
					MER_val[curr_epoch] = objective_function(model, x_val, y_val, n_per_sample_val, z_monitor_all_val)
					now = time.time()
					tput = float(size_epoch*monitor_frequency*batchsize) / (now-period_start_at)
					tpassed = now-start_at
					print "   %.1fs Epoch %d, batch %d, Probloss on Validation Set %.4f, %.2f S/s" % \
						(tpassed, curr_epoch, bi, MER_val[curr_epoch],tput)
					# Reset
					period_start_at = time.time()
				
				curr_epoch += 1
				if curr_epoch >= Nepoch:
					print("we're stopping")
					break
			bi += 1  # Batch index
			indexes = np.sort(np.random.choice(N, batchsize, replace=False))
			x = chainer.Variable(xp.asarray(x_train[indexes]).astype(xp.float32))
			y = chainer.Variable(xp.asarray(y_train[indexes]).astype(xp.float32))   
			# Reset/forward/backward/update
			opt_model.update(model, x, y)


		serializers.save_npz(savedir + '/model_end.model', model)
		np.savetxt(savedir + 'objective_values', [objective_values])
		np.savetxt(savedir + 'MER_val', [MER_val])
		serializers.save_npz(savedir + 'optimizer_end.state', opt_model)
