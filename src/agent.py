import torch

from brain import Brain


class Agent:

    def __init__(self, genome):

        self.x = 0.0
        self.y = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.theta = 0.0
        self.omega = 0.0

        #self.h = torch.zeros(16)

        self.color = [1.0, 0.0, 0.0]

        self.energy = 0.0

        self.brain = Brain(genome)


    def get_input(self, env):

        # Sensor center in the agent's local coordinates
        sensor_center = torch.tensor([-10.0, 0.0])

        c = torch.cos(torch.tensor(self.theta))
        s = torch.sin(torch.tensor(self.theta))

        R = torch.tensor([
            [c, -s],
            [s,  c]
        ])

        sensor_center = R @ sensor_center
        sensor_center[0] += self.x
        sensor_center[1] += self.y

        inputs = torch.zeros(10, 10, 3)

        for i in range(10):
            for j in range(10):

                # Pixel coordinates relative to the sensor center
                local = torch.tensor([
                    i - 4.5,
                    j - 4.5
                ])

                world = R @ local
                world += sensor_center

                x = int(round(world[0].item()))
                y = int(round(world[1].item()))

                if 0 <= x < env.width and 0 <= y < env.height:
                    inputs[j, i] = env.field[y, x]

        return inputs.flatten()

    def get_polygon(self):

        # Rectangle centered at the origin
        corners = torch.tensor([
            [-5.0, -2.5],
            [ 5.0, -2.5],
            [ 5.0,  2.5],
            [-5.0,  2.5],
        ])

        c = torch.cos(torch.tensor(self.theta))
        s = torch.sin(torch.tensor(self.theta))

        R = torch.tensor([
            [c, -s],
            [s,  c]
        ])

        corners = corners @ R.T

        corners[:, 0] += self.x
        corners[:, 1] += self.y

        return corners

    def step(self, env, dt):

        # Sense
        inp = self.get_input(env)

        # Think
        output = self.brain(inp)

        # Motor neurons (non-negative firing)
        forward = output[0] - output[1]
        turn = output[2] - output[3]

        # Act
        # Turn

        self.omega += turn.item() * dt

        self.omega *= 0.1      # angular damping

        #self.theta += self.omega * dt
        self.theta += turn.item() * dt

        # Acceleration
        ax = torch.cos(torch.tensor(self.theta)) * forward
        ay = torch.sin(torch.tensor(self.theta)) * forward

        self.vx += ax.item() * dt
        self.vy += ay.item() * dt

        # Friction / drag
        friction = 0.9
        self.vx *= friction
        self.vy *= friction

        # Position
        self.x += self.vx * dt
        self.y += self.vy * dt


                # Left/right boundaries
        if self.x < 0:
            self.x = 0
            self.vx = 0.0

        elif self.x > env.width - 1:
            self.x = env.width - 1
            self.vx = 0.0

        # Top/bottom boundaries
        if self.y < 0:
            self.y = 0
            self.vy = 0.0

        elif self.y > env.height - 1:
            self.y = env.height - 1
            self.vy = 0.0

        env.consume_food(self)
