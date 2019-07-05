import gym
import odeworldgym
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2
import os.path
import time
import os
start = time.time()

env_name = os.environ['ENV']

env = gym.make(env_name)

# env = FrameStack(env, n_frames=3)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=3)
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('SubWorld-v0') for i in range(n_cpu)])

# policy_kwargs =  dict(net_arch=[256, 256])

if os.path.isfile("{}.pkl".format(env_name)):
	model = PPO2.load(env_name, env=env, verbose=1, tensorboard_log="./ppo_tensorboard/")
else:
	model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")
	# model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")

print("Training model.")

# Save model periodically
# for i in range(1000):
model.learn(total_timesteps=10000)
model.save(env_name)

env.close()

print("Total time taken:")
end = time.time()
print(end - start)
