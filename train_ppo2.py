import gym
import subworldgym
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
import os.path


env = gym.make('SubWorld-v0')
# env = FrameStack(env, n_frames=3)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=3)
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('SubWorld-v0') for i in range(n_cpu)])

policy_kwargs = dict(net_arch=[256, 256])

if os.path.isfile('submarine_ppo.pkl'):
	model = PPO2.load('submarine_ppo', env=env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./ppo_tensorboard/")
else:
	model = PPO2(CnnPolicy, env, verbose=1, policy_kwargs=policy_kwargs,tensorboard_log="./ppo_tensorboard/")
	# model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")

print("Training model.")

# Save model periodically
# for i in range(1000):
model.learn(total_timesteps=5000)
model.save("submarine_ppo")

env.close()