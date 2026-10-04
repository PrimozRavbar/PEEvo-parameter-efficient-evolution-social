
import subprocess
import torch

from run_episode import run_episode


def reproduce(survivors):

    offspring_counts = [6, 5, 4, 3, 1]

    new_genomes = []

    for parent, count in zip(survivors, offspring_counts):

        for i in range(count):

            child = parent.clone()

            # Keep first offspring identical, mutate the rest
            if i > 0:
                child.mutate()

            new_genomes.append(child)

    return new_genomes


def evolve(genomes, generations=200):

    initial_gen = 11

    hall_of_fame = []

    for g in range(initial_gen, generations):

        episode_fitness = [
            run_episode(genomes, steps=300)
            for _ in range(6)
        ]

        fitness = [
            sum(episode[i] for episode in episode_fitness) / 6
            for i in range(len(genomes))
        ]

        ranking = sorted(
            range(len(genomes)),
            key=lambda i: fitness[i],
            reverse=True
        )

        hall_of_fame.extend([
            (fitness[i], genomes[i].clone())
            for i in ranking[:2]
        ])

        hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        hall_of_fame = hall_of_fame[:20]

        survivors = [genomes[i] for i in ranking[:5]]

        genomes = reproduce(survivors)

        for i, (_, genome) in enumerate(hall_of_fame):
            if i < len(genomes):
                genomes[i] = genome.clone()

        print(
            f"Generation {g}: "
            f"best={max(fitness):.2f}, "
            f"avg={sum(fitness)/len(fitness):.2f}"
        )

        if (g + 1) % 5 == 0:

            checkpoint = f"checkpoint/generation_{g + 1}.pt"

            # Get any remote commits before creating the new checkpoint commit
            subprocess.run(
                ["git", "pull", "--rebase", "origin", "main"],
                check=True
            )

            torch.save(
                {
                    "genomes": genomes,
                    "hall_of_fame": hall_of_fame
                },
                checkpoint
            )

            subprocess.run(
                ["git", "add", checkpoint],
                check=True
            )

            subprocess.run(
                [
                    "git", "commit",
                    "-m",
                    f"Checkpoint generation {g + 1}"
                ],
                check=True
            )

            subprocess.run(
                ["git", "push", "origin", "main"],
                check=True
            )

            print(f"Checkpoint pushed: generation {g + 1}")

    return genomes, hall_of_fame
