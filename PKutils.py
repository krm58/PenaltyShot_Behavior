import sys
import csv
import numpy as np
import gpflow
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.cluster.vq import kmeans
tf.set_random_seed(1234)
import pickle

def loaddata(subID, whichModel='PSwitch'):
	'''
	Loads data for subject-level modeling
	'''
	data = h5py.File('penaltykickdata.h5','r')
	subID1HE = np.array(data.get('subID')).astype('float32')
	otherdata = np.array(data.get('otherfeatures')).astype('float32')
	switchBool = np.array(data.get('targets')).astype('float32') #did they switch at time t+1
	trialidx = np.array(data.get('trialidx')).astype('float32')
	time = np.array(data.get('time')).astype('int32')

	if whichModel == 'PSwitch':
	    targets = np.array(data.get('targets')).astype('float32')
	elif whichModel == 'ExtraEV':
	    targets = np.array(data.get('EVtargets').value).astype('int32')
	    otherdata = np.hstack((otherdata, switchBool))
	Xfeatures_totaldata = np.hstack((otherdata, subID1HE))
	Xfeatures_totaldata = pd.DataFrame(Xfeatures_totaldata)
	offset = otherdata.shape[1]
	subdata = Xfeatures_totaldata[Xfeatures_totaldata[offset+subID]==1]
	subtargets = pd.DataFrame(targets).iloc[subdata.index]
	X = pd.DataFrame(otherdata).iloc[subdata.index]

	#Make within-opponent experience percentage variable, but only for the PSwitch model (not EV)
	if whichModel == 'PSwitch':
		X["superindex"] = pd.DataFrame(trialidx).iloc[subdata.index]
		progressvar = []
		humantrialindex = X[X[5]==1]['superindex'].unique()
		cputrialindex = X[X[5]==0]['superindex'].unique()
		numHuman = len(humantrialindex)
		numCPU = len(cputrialindex)

		for _, row in X.iterrows():
			if row[5] == 1: #if human observation
				progressvar.append((np.where(humantrialindex == row['superindex'])[0][0] / numHuman))
			elif row[5] == 0: #if cpu observation
				progressvar.append((np.where(cputrialindex == row['superindex'])[0][0] / numCPU))
		X['progressvar'] = progressvar
		del X['superindex']

	X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
	y_train, y_test = train_test_split(subtargets, test_size=0.2, random_state=1)

	return X, subtargets, X_train, X_test, y_train, y_test

def loadGPmodel_PSwitch(subID, numIPs=500, iters=200000, mb=256, npseed=1):
    
	X, subtargets, X_train, X_test, y_train, y_test = loaddata(subID)

	np.random.seed(npseed)
	Ms = numIPs
	X = np.array(X_train, dtype=float)
	Y = np.array(y_train, dtype=float)
	Z = kmeans(X_train, Ms, iter=1)[0]
	Z = np.array(Z, dtype=float)
	dimsize = X.shape[1]
	kernel = gpflow.kernels.RBF(input_dim=dimsize, ARD=True)

	#to load in
	with open('finalindividsubjGPs/pswitchmodel_fulltrimdata_' + str(numIPs) + 'IP_sub' + str(subID) + '_np' + str(npseed) + '_iters' + str(iters) + '.pickle', 'rb') as handle:
		models = pickle.load(handle)

	m = gpflow.models.SVGP(X,Y, kern=kernel,likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, minibatch_size=mb)

	with open('finalindividsubjGPs/pswitchmodelparams_fulltrimdata_' + str(numIPs) + 'IP_sub' + str(subID) + '_np' + str(npseed) + '_iters' + str(iters), 'rb') as handle:
		model = pickle.load(handle)

	m.assign(model.value)

	return m

def loadGPmodel_EV(subID, numIPs=500, iters=200000, mb=256, npseed=1):

	X, subtargets, X_train, X_test, y_train, y_test = loaddata(subID,whichModel='ExtraEV')

	np.random.seed(npseed)
	Ms = numIPs
	X = np.array(X_train, dtype=float)
	Y = np.array(y_train, dtype=float)
	Z = kmeans(X_train, Ms, iter=1)[0]
	Z = np.array(Z, dtype=float)
	dimsize = X.shape[1]
	kernel = gpflow.kernels.RBF(input_dim=dimsize, ARD=True)

	#to load in
	with open('ExtraEVfinalindividsubjGPs/fulltrimdata_' + str(numIPs) + 'IP_sub' + str(subID) + '_np' + str(npseed) + '_iters' + str(iters) + '.pickle', 'rb') as handle:
		models = pickle.load(handle)

	m = gpflow.models.SVGP(X,Y, kern=kernel, likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, minibatch_size=mb)

	with open('ExtraEVfinalindividsubjGPs/params_fulltrimdata_' + str(numIPs) + 'IP_sub' + str(subID) + '_np' + str(npseed) + '_iters' + str(iters), 'rb') as handle:
		model = pickle.load(handle)
	m.assign(model.value)

	return m

def calculateProbSwitch(subID):

	inputdata, subtargets, X_train, X_test, y_train, y_test = loaddata(subID, whichModel='PSwitch')
	m = loadGPmodel_PSwitch(subID, numIPs=500, iters=200000, mb=256, npseed=1)

	probs = []
	Xfeatures_totaldata = np.array(inputdata, dtype=float)    
	dataset = tf.contrib.data.Dataset.from_tensor_slices(Xfeatures_totaldata)
	dataset = dataset.batch(len(inputdata))
	iterator = dataset.make_one_shot_iterator()
	data = iterator.get_next()
	m.initialize()   

	with tf.Session() as sess:
		probs = m.predict_y(data.eval())[0]

	return probs

def calculateExtraEV(subID):
	"""
	calculate the EV from the observed outcome
	"""

	inputdata, subtargets, X_train, X_test, y_train, y_test = loaddata(subID,whichModel='ExtraEV')
	m = loadGPmodel_EV(subID, numIPs=500, iters=200000, mb=256, npseed=1)
	EVs = []

	Xfeatures_totaldata = np.array(inputdata, dtype=float)    
	dataset = tf.contrib.data.Dataset.from_tensor_slices(Xfeatures_totaldata)
	dataset = dataset.batch(len(inputdata))
	iterator = dataset.make_one_shot_iterator()
	data = iterator.get_next()
	m.initialize()   

	with tf.Session() as sess:
		EVprobs = m.predict_y(data.eval())[0]
	return EVprobs

def kelsey_calc_whitened_indices(m, X, inds):
	"Whitened"
	f = m.q_mu._constrained_tensor
	Z = m.feature.Z._constrained_tensor
	K = m.kern.K(Z) + tf.eye(tf.shape(Z)[0], 
	                         dtype=gpflow.settings.float_type) * gpflow.settings.numerics.jitter_level
	LK = tf.cholesky(K)
	lenscales = tf.gather(m.kern.lengthscales._constrained_tensor, inds)

	kvec = m.kern.K(X, Z)
	kscal = m.kern.variance.constrained_tensor
	dX = (tf.expand_dims(tf.gather(Z, inds, axis=1), 0) - tf.expand_dims(tf.gather(X, inds, axis=1), 1))
	dk = lenscales**(-2) * dX * tf.expand_dims(kvec, 2)

	# first piece of covariance
	ddk = tf.diag(kscal * lenscales**(-2))

	# second piece of covariance
	LKinvdk = tf.stack(tf.map_fn(lambda x: tf.matrix_triangular_solve(LK, x, lower=True), dk))
	dkKinvdk = tf.matmul(LKinvdk, LKinvdk, transpose_a=True)

	# mean
	dmu = tf.einsum('bmd,mj->bd', LKinvdk, f)

	# put it all together and invert
	dSigma = (ddk - dkKinvdk)
	L = tf.cholesky(dSigma)

	df_white = tf.matrix_triangular_solve(L, tf.expand_dims(dmu,2), lower=True)
	return df_white

def calculateExtraEV_withcounter(subID, manip = 'real'):
	"""
	Manip variable is a string that takes on one of three strings:
	--"real" means calculate the EV from the observed outcome
	--"all1" means change all currswitch input vars to 1, then calculate EV
	--"all0" means change all currswitch input vars to 0, then calculate EV
	"""

	inputdata, subtargets, X_train, X_test, y_train, y_test = loaddata(subID,whichModel='ExtraEV')
	m = loadGPmodel_EV(subID, whichModel='ExtraEV', numIPs=500, iters=200000, mb=256, npseed=1)

	EVs = []
    
	if manip == 'all1':
		inputdata[7] = 1
	elif manip == 'all0':
		inputdata[7] = 0

	Xfeatures_totaldata = np.array(inputdata, dtype=float)    
	dataset = tf.contrib.data.Dataset.from_tensor_slices(Xfeatures_totaldata)
	dataset = dataset.batch(len(inputdata))
	iterator = dataset.make_one_shot_iterator()
	data = iterator.get_next()
	m.initialize()   

	with tf.Session() as sess:
		EVprobs = m.predict_y(data.eval())[0]

	return EVprobs

def loadshootersensmetric(subID):
	return np.load("finalindividsubjGPs/pswitchgradmetric_trimsub{}_500IPs_npseed1_200000iters.npy".format(subID))

def loaddf():
	'''
    Load the preprocessed data for the behavioral analysis
	'''
	data = h5py.File('penaltykickdata.h5','r')
	subID1HE = np.array(data.get('subID')).astype('float32')
	otherdata = np.array(data.get('otherfeatures')).astype('float32')
	switchBool = np.array(data.get('targets')).astype('float32') #did they switch at time t+1
	trialidx = np.array(data.get('trialidx')).astype('float32')
	time = np.array(data.get('time')).astype('int32')
	policytargets = np.array(data.get('targets')).astype('float32')
	EVtargets = np.array(data.get('EVtargets').value).astype('int32')
	subIDs = pd.DataFrame(subID1HE).idxmax(axis=1).values
	df = pd.DataFrame(otherdata)
	df.rename(index=str, columns={0:"goalieypos",1: "ball_xpos",2:"ball_ypos",3: "goalie_yvel",4: "ball_yvel",5: "opp",6: "tslc"},inplace=True)
	df['super_index'] = trialidx
	df['time'] = time
	df['subID'] = subIDs
	df['result'] = EVtargets
	df['shooterswitches'] = policytargets
	return df
