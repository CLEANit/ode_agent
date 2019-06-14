import gym
import subworldgym
import numpy as np
from time import sleep
import numpy as np
# from stable_baselines.common.atari_wrappers import FrameStack

# Demo for using SubWorld-v0

# Initialize the environment
env = gym.make('SubWorld-v0')
# env = FrameStack(env, n_frames=3)
# Mode for rendering demo ('human' = normal rendering, 'full' = full information rendering, 'save' = save normal rendering figures in 'SUBworld/subworldgym/figures/SubWorldv0/', 'save_full' = save full rendering figures in 'SUBworld/subworldgym/figures/SubWorldv0/')
render_mode = 'human'

env.reset()
# Reset the environment to get initial state
env.unwrapped.render(mode = render_mode)
#wait = input('PRESS ENTER TO CONTINUE.')

d = False
i = 0
speed = 1.0

# Game will play until: 100 steps are reached, or the sub crashes into an island
# while not d:
while True:
    # Select a random action
    a = env.action_space.sample()

    # a[1] = 0.0
    speed -= 0.1 
    speed = max(speed, 0.0)
    a[0] = speed
    print(a)

    s, r, d, _ = env.step(a)

    # print(s.shape)
    s = np.array(s)
    # np.savetxt('a.csv', s[:, : , 0])



    env.unwrapped.render(mode = 'full')
    i += 1
    print(i, r, d)
    if d or i > 20:
        speed = 1.0
        
    if i > 100:
        i = 0
        s = env.reset()

wait = input('PRESS ENTER TO EXIT.')

