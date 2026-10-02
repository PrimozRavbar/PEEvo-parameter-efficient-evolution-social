import torch
import imageio.v3 as iio

from environment import Environment
from agent import Agent


def run_episode(
    genomes,
    steps=500,
    render=False,
    movement="continuous"
):

    env = Environment()

    agents = []

    for genome in genomes:

        agent = Agent(
            genome,
            movement=movement
        )

        agent.x = torch.randint(
            0,
            env.width,
            (1,)
        ).item()

        agent.y = torch.randint(
            0,
            env.height,
            (1,)
        ).item()

        agent.theta = (
            2 * torch.pi * torch.rand(1).item()
        )

        agents.append(agent)

    frames = []

    for t in range(steps):

        env.update(agents)

        for agent in agents:
            agent.step(env, dt=1.0)

        if render:

            frames.append(
                (255 * env.field)
                .clamp(0, 255)
                .byte()
                .numpy()
            )

    if render:

        iio.imwrite(
            "simulation.mp4",
            frames,
            fps=20
        )

    fitness = [
        agent.energy
        for agent in agents
    ]

    return fitness
