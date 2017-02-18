#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_dagger_policy.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

import imit_policy as imit
import parameter as par

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    
    with open('dagger_database/'+args.envname, 'rb') as f:
        data = pickle.load(f)
        tempx=data['observations']
        temp=tempx.shape
        nin=temp[1]
        tempy=data['actions']
        temp=tempy.shape
        nout=temp[2]
    
    policy_expert = load_policy.load_policy(args.expert_policy_file)
    x, y = imit.placeholder_inputs(None, nin, nout, par.batch_size)
    policy_fn = imit.inference(x, nin, nout, par.n_h1, par.n_h2, par.n_h3)
    saver = tf.train.Saver()
    print('loaded and built')

    #init = tf.global_variables_initializer()
    with tf.Session():
        #tf_util.get_session().run(init)
        tf_util.initialize()
        saver.restore(tf_util.get_session(), "trainedNN/"+args.envname)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        actions_expert = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action_expert = policy_expert(obs[None,:])
                action = tf_util.get_session().run([policy_fn],feed_dict={x:obs[None,:]})
                observations.append(obs)
                actions.append(action)
                actions_expert.append(action_expert)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        '''
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        '''
        expert_data = {'observations': np.concatenate((tempx,
                                                       np.array(observations))),
                       'actions': np.concatenate((tempy,
                                                  np.array(actions_expert)))}
        # save expert policy observations
        with open('dagger_database/'+args.envname, 'wb') as f:
            pickle.dump(expert_data, f)

if __name__ == '__main__':
    main()
