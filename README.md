# PenaltyShot_Behavior
Repository for analysis code used in the Penalty Shot task, published in Nature Communications.

## Data
Preprocessed data for the Penalty Shot behavioral task can be found on Open Science Framework (https://osf.io/evfg5/). `penaltykickdata.h5` is the name of the dataset. This should be downloaded and included in the `PenaltyShot_Behavior` path before continuing. 

## Types of Models Included
There are three main classes of models included in the Penalty Shot behavioral manuscript: Policy Models, Value Function Models, and Final Shot Models. 
  - Policy Models provide a mapping of states to actions (in other words, they predict the probability of conducting a given action given the state of the environment). In the case of our paradigm, given the state of the game, policy models predict the likelihood of the human subject changing vertical direction (i.e. changing his/her trajectory from up to down or vice versa). These Gaussian Process Policy models are individualized, so there is 1 Gaussian Process model per subject (82 total). Policy models from 4 subjects are included in this repository for illustration purposes. If you'd like access to all 82 subjects' policy gaussian process models, these can be generated from TrainGPFlow.py. 
  - Value Function Models predict the likelihood of a given subject winning the trial, based on the instantaneous state of the game. Like the policy models, each subject has his/her own value function model (82 total).
  - Final Shot Models predict the likelihood of winning \emph{given} that the input data correspond to a player's final change in direction. Because the input data is characterized as being the last action the subject takes, it follows that the likelihood of winning after this point is determined on the policy of the goalie. Thus, whereas each subject had his/her own policy and value function models, there are three final shot models, corresponding to one per goalie (there were two human goalies used throughout the course of data collection, but each subject only interacted with one). Thus, Human Goalie 1, Human Goalie 2, and the computer goalie each have his/her own final shot Gaussian Process model. 
  
## Training the Models
All three final shot models are included in the  `VnoswitchGPs` folder. The Policy and Value Function models from four subjects are included in `finalindividsubjGPs` and `ExtraEVfinalindividsubjGPs`, respectively. For training every subjects' models from scratch, this can be done by running `TrainIndividGPFlow.py`, where subID is an integer from 0 to 81 corresponding to the subject whose data you wish the model. For training a subject's policy model, `--whichModel` should be entered to be `PSwitch`; for training a subject's value function model, it should be `ExtraEV`. For replicating figures in the paper, the authors used 500 inducing points (`--IP`) and 2000,000 iterations. Please note, however, that results might differ slightly despite setting the numpy and tensorflow seeds due to certain operations in tensorflow that are nondeterministic. Training the Final Shot models from scratch can be done with `TrainHumanGoalie1LastShooterSwitch.py`, `TrainHumanGoalie2LastShooterSwitch.py`, and `TrainCPULastShooterSwitch.py`. 

## Analyzing the Trained Models
For analyzing models that have been trained and stored in `finalindividsubjGPs`, `ExtraEVfinalindividsubjGPs`, and `VnoswitchGPs`, depending on the model being trained, please turn to `GP_PSwitch_Analysis.ipynb` for analyzing the Policy models, `GP_ExtraEVAnalyses.ipynb` for analyzing the Value Function models, and `FinalMoveAnalysis.ipynb` for analyzing the final shot models. All of these notebooks will use `PKutils.py` to load in helper functions.

## Further Questions
If you have questions regarding running parts of the code base, or have questions about the analysis pipeline we used in general, please contact <krm58@duke.edu>.
