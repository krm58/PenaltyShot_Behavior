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
import PKutils

def train_model(**kwargs):
	npseed = kwargs['npseed']
	iters = kwargs['iterations']
	gpu = kwargs['gpu']
	numInducingPoints = kwargs['IP']
	dataversion = kwargs['dataversion']
	print("npseed: " + str(npseed))
	print("iterations: " + str(iters))
	print("gpu: " + str(gpu))
	print("IPs: " + str(numInducingPoints))

	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

	if dataversion == 'behavioral':
		print("Loading Data (behavioral)....")
		Vnoswitchdf = pd.read_csv("LastSwitchThatTrial.csv")
	elif dataversion == 'neuroimaging':
		print("Loading version of behavioral data for neuroimaging (sub 50's run 2,3,4)")
		Vnoswitchdf = pd.read_csv("LastSwitchThatTrial_neuroimaging.csv")
	
	del Vnoswitchdf["Unnamed: 0"]
	result = Vnoswitchdf["result"]
	testtrim = Vnoswitchdf[["goalieypos","ball_xpos","ball_ypos","goalie_yvel","ball_yvel","opp","tslc","subID"]]
	cputrialsdf = testtrim[testtrim["opponent"]==0] #trials against computer goalie
	cputrialsdf_result = result.loc[cputrialsdf.index]
	humantrialsdf = testtrim[testtrim["opponent"]==1]
	humantrialsdf_result = result.loc[humantrialsdf.index]
	humantrialsdf["subID"] = humantrialsdf["subID"].astype('int')
	goalie1trialsdf = humantrialsdf[humantrialsdf["subID"]<50]
	goalie1trialsdf_result = humantrialsdf_result.loc[goalie1trialsdf.index]
	goalie2trialsdf = humantrialsdf[humantrialsdf["subID"]>=50]
	goalie2trialsdf_result = humantrialsdf_result.loc[goalie2trialsdf.index]
	del goalie2trialsdf["subID"]
	del goalie1trialsdf["subID"]
	del cputrialsdf["subID"]

	# Train the GPs
	X_train, X_test = train_test_split(goalie2trialsdf, test_size=0.2, random_state=1)
	y_train, y_test = train_test_split(goalie2trialsdf_result, test_size=0.2, random_state=1)
	optimizer = 'Adam'
	mb = 256

	np.random.seed(npseed)
	Ms = numInducingPoints
	X = np.array(X_train, dtype=float) 
	Y = np.array(y_train, dtype=float)
	Y = np.expand_dims(Y,axis=-1)
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

	experstring = 'Vnoswitch_goalie2_iters' + str(iters) + '_inducingpts' + str(numInducingPoints) + '_' + "_npseed" + str(npseed)
	fw = tf.summary.FileWriter("Vnoswitchtrain_logs/{}".format(experstring), m.graph)

	#define summary scalars for examination in tensorboard
	tf.summary.scalar("likelihood", m._build_likelihood())
	tf.summary.scalar("lengthscales_goalieposy", tf.gather(m.kern.lengthscales._constrained_tensor, 0))
	tf.summary.scalar("lengthscales_shooterposx", tf.gather(m.kern.lengthscales._constrained_tensor, 1))
	tf.summary.scalar("lengthscales_shooterposy", tf.gather(m.kern.lengthscales._constrained_tensor, 2))
	tf.summary.scalar("lengthscales_goalievely", tf.gather(m.kern.lengthscales._constrained_tensor, 3))
	tf.summary.scalar("lengthscales_shootervely", tf.gather(m.kern.lengthscales._constrained_tensor, 4))
	tf.summary.scalar("lengthscales_opp", tf.gather(m.kern.lengthscales._constrained_tensor, 5))
	tf.summary.scalar("lengthscales_timesincelastchange", tf.gather(m.kern.lengthscales._constrained_tensor, 6))

	mysum = tf.summary.merge_all()
	def loss_callback(summary):
	    fw.add_summary(summary, loss_callback.iter)
	    loss_callback.iter += 1
	loss_callback.iter=0
	print("Training goalie2 Value Model...")
	gpflow.train.AdamOptimizer(learning_rate).minimize(m, maxiter=iters, var_list=[global_step], global_step=global_step, summary_op=mysum, file_writer=fw)

	#save model
	param_dict = {p[0].full_name.replace('SGPR', 'SGPU'): p[1] for p in zip(m.trainable_parameters, m.read_trainables())}
	

	with open('VnoswitchGPs/noswitchVmodel_'+str(numInducingPoints)+'IP_np'+str(npseed)+ '_iters' + str(iters) + '.pickle', 'wb') as handle:
	    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	m.as_pandas_table().to_pickle('VnoswitchGPs/modelparams_'+str(numInducingPoints)+'IP_np'+str(npseed) + '_iters'+str(iters))

	print("goalie2 Value GP Complete")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--npseed',default=1, type=int)
	parser.add_argument('--iterations',default=100000,type=int)
	parser.add_argument('--gpu',default=0, type=int)
	parser.add_argument('--IP', default=500, type=int)
	parser.add_argument('--dataversion', type=str, default='behavioral') #either behavioral or neuroimaging
	args = parser.parse_args()
	train_model(**vars(args))