import gym
import subworldgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
import os.path

n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('SubWorld-v0') for i in range(n_cpu)])


# if os.path.isfile('submarine.pkl'):
# 	model = PPO2.load('submarine_parallel', env=env, verbose=1, tensorboard_log="./ppo_parallel_tensorboard/")
# else:
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")

print("Training model.")
model.learn(total_timesteps=10000)

print("Saving the model.")
model.save("submarine_ppo")
env.close()