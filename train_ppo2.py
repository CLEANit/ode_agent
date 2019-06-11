import gym
import subworldgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import os.path

env = DummyVecEnv([lambda: gym.make('SubWorld-v0')])

if os.path.isfile('submarine_ppo.pkl'):
	model = PPO2.load('submarine_ppo', env=env, verbose=1, tensorboard_log="./ppo_tensorboard/")
else:
	model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")

print("Training model.")

# Save model periodically
for i in range(1000):
	model.learn(total_timesteps=1000)
	model.save("submarine_ppo")

env.close()