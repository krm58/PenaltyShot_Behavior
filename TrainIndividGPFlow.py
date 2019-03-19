import sys
import csv
import numpy as np
import gpflow
import os
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy.cluster.vq import kmeans
tf.set_random_seed(1234)
import pickle
import argparse

#Model to train the individual policy models for Penalty Shot

def train_model(**kwargs):
	subID = kwargs['subID']
	npseed = kwargs['npseed']
	iters = kwargs['iterations']
	gpu = kwargs['gpu']
	numInducingPoints = kwargs['IP']
	whichModel = kwargs['whichModel']
	print("subID: " + str(subID))
	print("npseed: " + str(npseed))
	print("iterations: " + str(iters))
	print("gpu: " + str(gpu))
	print("IPs: " + str(numInducingPoints))
	print("Model Requested: " + str(whichModel))

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

	print("Loading Data for Subject " + str(subID) + "....")

	if subID == 50: #this sub had 4 runs, 1 with fatsat, 2,3,4 without fatsat
		data = h5py.File('penaltykickdata_neuroimaging_234fixed.h5')
		print("Using the data version of Runs 2,3,4 without fatsat!")
	else:
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
	
	#Make within-opponent experience percentage variable, but only for the PSwitch model (not ExtraEV) because this is confounded
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

	optimizer = 'Adam'
	mb = 256

	np.random.seed(npseed)
	Ms = numInducingPoints
	X = np.array(X_train, dtype=float) 
	Y = np.array(y_train, dtype=float)
	Z = kmeans(X_train, Ms, iter=1)[0]
	Z = np.array(Z, dtype=float)
	dimsize = X.shape[1]
	kernel = gpflow.kernels.RBF(input_dim=dimsize, ARD=True)

	m = gpflow.models.SVGP(
	    X,Y, kern=kernel,
	    likelihood=gpflow.likelihoods.Bernoulli(), Z=Z, minibatch_size=mb)
	m.feature.set_trainable(True)

	global_step = tf.get_variable("global_step", (), tf.int32, tf.zeros_initializer(), trainable=False)
	learning_rate = 0.001 #adam default


	if whichModel == 'PSwitch':
		if subID != 50:
			experstring = 'fulldatatrim_sub' + str(subID) + '_iters' + str(iters) + '_inducingpts' + str(numInducingPoints) + '_' + "_npseed" + str(npseed)
		else:
			experstring = 'runs234_sub' + str(subID) + '_iters' + str(iters) + '_inducingpts' + str(numInducingPoints) + '_' + "_npseed" + str(npseed)
		fw = tf.summary.FileWriter("finalsubjtrainindivid_logs/{}".format(experstring), m.graph)
	elif whichModel == 'ExtraEV':
		if subID != 50:
			experstring = 'ExtraEV_sub' + str(subID) + '_iters' + str(iters) + '_inducingpts' + str(numInducingPoints) + '_' + "_npseed" + str(npseed)	
		else:
			experstring = 'ExtraEV_runs234_sub' + str(subID) + '_iters' + str(iters) + '_inducingpts' + str(numInducingPoints) + '_' + "_npseed" + str(npseed)
		fw = tf.summary.FileWriter("ExtraEVfinalsubjtrainindivid_logs/{}".format(experstring), m.graph)

	#define summary scalars for examination in tensorboard
	tf.summary.scalar("likelihood", m._build_likelihood())
	tf.summary.scalar("ELBO", m.likelihood_tensor)
	tf.summary.scalar("lengthscales_goalieposy", tf.gather(m.kern.lengthscales._constrained_tensor, 0))
	tf.summary.scalar("lengthscales_shooterposx", tf.gather(m.kern.lengthscales._constrained_tensor, 1))
	tf.summary.scalar("lengthscales_shooterposy", tf.gather(m.kern.lengthscales._constrained_tensor, 2))
	tf.summary.scalar("lengthscales_goalievely", tf.gather(m.kern.lengthscales._constrained_tensor, 3))
	tf.summary.scalar("lengthscales_shootervely", tf.gather(m.kern.lengthscales._constrained_tensor, 4))

	tf.summary.scalar("lengthscales_opp", tf.gather(m.kern.lengthscales._constrained_tensor, 5))
	tf.summary.scalar("lengthscales_timesincelastchange", tf.gather(m.kern.lengthscales._constrained_tensor, 6))
	if whichModel == 'PSwitch':
		tf.summary.scalar("lengthscales_oppexperience", tf.gather(m.kern.lengthscales._constrained_tensor, 7))
	elif whichModel == 'ExtraEV':
		tf.summary.scalar("lengthscales_currtswitch", tf.gather(m.kern.lengthscales._constrained_tensor,7))

	mysum = tf.summary.merge_all()
	def loss_callback(summary):
	    fw.add_summary(summary, loss_callback.iter)
	    loss_callback.iter += 1
	loss_callback.iter=0

	print("Training Model...")
	gpflow.train.AdamOptimizer(learning_rate).minimize(m, maxiter=iters, var_list=[global_step], global_step=global_step, summary_op=mysum, file_writer=fw)

	#save model
	param_dict = {p[0].full_name.replace('SGPR', 'SGPU'): p[1] for p in zip(m.trainable_parameters, m.read_trainables())}
	
	if whichModel == 'PSwitch':
		if subID != 50:
			with open('finalindividsubjGPs/pswitchmodel_fulltrimdata_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed)+ '_iters' + str(iters) + '.pickle', 'wb') as handle:
				pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			m.as_pandas_table().to_pickle('finalindividsubjGPs/pswitchmodelparams_fulltrimdata_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed) + '_iters'+str(iters))
		else:
			with open('finalindividsubjGPs/pswitchmodel_dataruns234_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed)+ '_iters' + str(iters) + '.pickle', 'wb') as handle:
				pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			m.as_pandas_table().to_pickle('finalindividsubjGPs/pswitchmodelparams_dataruns234_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed) + '_iters'+str(iters))
	elif whichModel == 'ExtraEV':
		if subID != 50:
			with open('ExtraEVfinalindividsubjGPs/fulltrimdata_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed)+ '_iters' + str(iters) + '.pickle', 'wb') as handle:
				pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			m.as_pandas_table().to_pickle('ExtraEVfinalindividsubjGPs/params_fulltrimdata_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed) + '_iters'+str(iters))
		else:
			with open('ExtraEVfinalindividsubjGPs/dataruns234'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed)+ '_iters' + str(iters) + '.pickle', 'wb') as handle:
				pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
			m.as_pandas_table().to_pickle('ExtraEVfinalindividsubjGPs/params_dataruns234_'+str(numInducingPoints)+'IP_sub'+str(subID)+'_np'+str(npseed) + '_iters'+str(iters))


	print("Calculating Goalie Gradient Metric....")
	
	inds = np.array([0, 3], dtype=np.int32) #for goalie position and goalie y-velocity variables
	Xph = tf.placeholder(m.X.dtype, shape=(inputdata.shape[0], inputdata.shape[1]))
	data = np.array(inputdata, dtype=m.X.dtype)
	fd = {Xph: data}

	Xdata = tf.data.Dataset.from_tensor_slices(Xph).batch(mb)
	iterator = Xdata.make_initializable_iterator()
	next_element = iterator.get_next()
	df_white = kelsey_calc_whitened_indices(m, next_element, inds)
	dfs = []

	with tf.Session() as sess:
	    m.initialize()
	    sess.run(iterator.initializer, feed_dict=fd)
	    while True:
	          try:
	            dfs.append(sess.run([df_white], feed_dict=fd)[0])
	          except tf.errors.OutOfRangeError:
	            break

	result = np.concatenate(dfs).squeeze()
	mywhitemetric = result[:,0]**2 + result[:,1]**2

	if whichModel == 'PSwitch':
		if subID != 50:
			np.save("finalindividsubjGPs/pswitchgradmetric_trimsub" + str(subID) + "_" + str(numInducingPoints) + "IPs_" + "npseed" + str(npseed) + '_' + str(iters) + 'iters.npy', mywhitemetric)
		else:
			np.save("finalindividsubjGPs/pswitchgradmetric_runs234_sub" + str(subID) + "_" + str(numInducingPoints) + "IPs_" + "npseed" + str(npseed) + '_' + str(iters) + 'iters.npy', mywhitemetric)

	elif whichModel == 'ExtraEV':
		if subID != 50:
			np.save("ExtraEVfinalindividsubjGPs/gradmetric_trimsub" + str(subID) + "_" + str(numInducingPoints) + "IPs_" + "npseed" + str(npseed) + '_' + str(iters) + 'iters.npy', mywhitemetric)
		else:
			np.save("ExtraEVfinalindividsubjGPs/gradmetric_runs234_sub" + str(subID) + "_" + str(numInducingPoints) + "IPs_" + "npseed" + str(npseed) + '_' + str(iters) + 'iters.npy', mywhitemetric)

		
	print("Subject " + str(subID) + " Complete")
	
def kelsey_calc_whitened_indices(m, X, inds):
    f = m.q_mu._constrained_tensor
    Z = m.feature.Z._constrained_tensor
    K = m.kern.K(Z) + tf.eye(tf.shape(Z)[0], 
                             dtype=gpflow.settings.float_type) * gpflow.settings.numerics.jitter_level
    LK = tf.cholesky(K)
    lenscales = tf.gather(m.kern.lengthscales._constrained_tensor, inds)

    kvec = m.kern.K(X, Z)
    kscal = m.kern.variance.constrained_tensor
    dX = (tf.expand_dims(tf.gather(Z, inds, axis=1), 0) 
         - tf.expand_dims(tf.gather(X, inds, axis=1), 1))
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--subID', type=int)
	parser.add_argument('--npseed',default=1, type=int)
	parser.add_argument('--iterations',default=200000,type=int)
	parser.add_argument('--gpu',default=0, type=int)
	parser.add_argument('--IP', default=500, type=int)
	parser.add_argument('--whichModel', default='PSwitch', type=str) #PSwitch or EV or ExtraEV
	args = parser.parse_args()
	train_model(**vars(args))