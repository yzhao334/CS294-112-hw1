# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

Several files are used to perform imitation learning task, including:
* imit_policy.py: basic structure of a 3-layer neural network
* train_imit_policy.py: use tensorflow to train an imitation policy
* run_imit_policy.py: show the performance of learned policy
* train_dagger_policy.py: train imitation policy, use database from DAGGER
* run_dagger_policy.py: run the learned policy and augment data set for DAGGER
* main_dagger.py: script that keeps performing policy and training policy

To show the performance of the learned policy, the file 'run_imit_policy.py' can always be used. To show the expert policy, use file 'run_expert.py'.
