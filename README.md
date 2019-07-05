# ode_agent

Supply the name of the enviornment inside the `ENV` environment variable.

To see how a random agent performs on the environment:

    ENV=ODEWorld_0-v0 python train_ppo2.py

To train:

    ENV=ODEWorld_0-v0 python train_ppo2.py

To run the trained agent:

    ENV=ODEWorld_0-v0 python run_ppo2.py
