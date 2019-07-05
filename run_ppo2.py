import gym
import odeworldgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from itertools import count
from stable_baselines.common.atari_wrappers import FrameStack
from stable_baselines.common.vec_env import VecFrameStack
import os

env_name = os.environ['ENV']

env = gym.make(env_name)
# env = FrameStack(env, n_frames=3)
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecFrameStack(env, n_stack=3)

model = PPO2.load(env_name)
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

    env.render(mode='full')

    if done:
        print("Game over. Total Reward: {}".format(r_total))
        r_total = 0.0
        

        print("Enter:")
        print("ENTER to quit.")
        print("C to continue.")
        press = input('>> ')
        if press != "C":
            break
        print("HERE")

        obs = env.reset()


    
    
env.close()

