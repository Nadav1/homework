import tensorflow as tf
import numpy as np
import argparse
import pickle
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import load_policy
import tf_util
import gym


def create_model(num_inputs, num_logits):
  model = Sequential()
  model.add(Dense(units=256, activation='tanh', input_dim=num_inputs))
  model.add(Dense(units=128, activation='tanh'))
  model.add(Dense(units=num_logits, activation='tanh'))

  model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
  return  model



def main():
  # parser = argparse.ArgumentParser()
  # parser.add_argument('--expert_data', type=str)
  # args = parser.parse_args()

  #expert_data = {"Hopper-v2", "Humanoid-v2"}
  expert_data = "Hopper-v2"
  num_rollouts = 10
  num_epochs = 6
  DAgger_iterations = 50
  render = True

  print ("Running task " + expert_data)

  # Init environment
  env = gym.make(expert_data)

  #Load expert observations and actions, create model (policy)
  policy, observations, actions = get_policy(expert_data)

  # Load Expert Policy
  expert_policy = load_policy.load_policy(os.path.join('experts',expert_data + ".pkl"))

  with tf.Session():
    tf_util.initialize()

    for dagger_it in range (DAgger_iterations):
      print ("DAgger iteration " + str(dagger_it))

      #Update policy according to observations and actions
      print ("Training Model... Iter: " + str(dagger_it))
      policy.fit(observations, actions, epochs=num_epochs, batch_size=100, verbose=2)

      #Run num_rollouts on env using this policy - get observations
      print ("Evaluating policy on environment... Iter: " + str(dagger_it))
      new_observations = run_policy(policy, num_rollouts, render, env)
      observations = np.concatenate((observations,new_observations[0]))

      #Get ideal actions from expert for above observations
      print("Quering expert for ideal actions... Iter: " + str(dagger_it))
      new_actions = get_expert_feedback(new_observations[0], expert_policy)
      actions = np.concatenate((actions,new_actions))




def get_policy(expert_name):
  with open(os.path.join('expert_data', expert_name + '.pkl'), 'rb') as f:
    expert_data = pickle.load(f)

  observations = expert_data['observations']
  actions = expert_data['actions'].squeeze()

  num_possible_actions = actions.shape[1]
  num_inputs = observations.shape[1]
  model = create_model(num_inputs=num_inputs, num_logits=num_possible_actions)

  return model, observations, actions



def run_policy(policy, num_rollouts, render, env):

  def policy_fn(model, input):
    action = model.predict(input)
    return action

  max_steps = env.spec.timestep_limit
  returns = []
  observations = []
  actions = []
  for i in range(num_rollouts):
    #if (i%10 ==0):
      #print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
      action = policy_fn(policy, obs[None, :])
      observations.append(obs)
      actions.append(action)
      obs, r, done, _ = env.step(action)
      totalr += r
      steps += 1
      if render:
        env.render()
      if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
      if steps >= max_steps:
        break
    returns.append(totalr)

  #print('returns', returns)
  print('mean return', np.mean(returns))
  print('std of return', np.std(returns))

  # actions = np.asarray(actions).squeeze()
  observations = np.asarray(observations)
  return  observations, actions



def get_expert_feedback(observations, expert_policy):
  actions = []
  for i in range (observations.shape[0]):
    obs = observations[i,:]
    action = expert_policy(obs[None,:])
    actions.append(action)
  return np.asarray(actions).squeeze()


if __name__ == '__main__':
    main()
