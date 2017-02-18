#!/usr/bin/env python
import subprocess

# run DAGGER, keep augmenting dataset and training

def main(pars):
    for i in range(5):
        # run trained policy and enlarge database
        procrun=subprocess.Popen("python run_dagger_policy.py experts/"+pars.envname+".pkl "+pars.envname+" --num_rollouts 20", shell=True)
        try:
            procrun.communicate()
        except subprocess.TimeoutExpired:
            kill(procrun.pid)
        proctrain=subprocess.Popen("python train_dagger_policy.py "+pars.envname+" --num_epoches 200 --firsttime 0 --step_size 1e-4", shell=True)# step_size 1e-3 to 1e-n decrease if not converging
        try:
            proctrain.communicate()
        except subprocess.TimeoutExpired:
            kill(proctrain.pid)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    main(args)
