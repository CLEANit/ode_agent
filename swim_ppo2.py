import gym
import subworldgym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from itertools import count
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.vec_env import VecFrameStack

env = gym.make('SubWorld-v0')
# env = FrameStack(env, n_frames=3)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecFrameStack(env, n_stack=3)

model = PPO2.load('submarine_ppo')
# model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log="./ppo_tensorboard/")

obs = env.reset()
done = False
r_total = 0.0

for i in count():
    action, _states = model.predict(obs)
    
    print(action)
    obs, [reward], [done], info = env.step(action)
    print(reward)
    r_total += reward

    env.render(mode='human')

    if done:
    	print("Game over. Total Reward: {}".format(r_total))
    	r_total = 0.0
    	obs = env.reset()

    if i > 200:
    	break
env.close()

