import gym
import subworldgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from itertools import count

env = gym.make('SubWorld-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2.load('submarine_ppo')

obs = env.reset()
done = False
r_total = 0.0

for i in count():
    action, _states = model.predict(obs)
    obs, [reward], [done], info = env.step(action)
    print(reward)
    r_total += reward

    env.render(mode='human')

    if done:
    	print("Game over. Total Reward: {}".format(r_total))
    	r_total = 0.0
    	obs = env.reset()

env.close()

