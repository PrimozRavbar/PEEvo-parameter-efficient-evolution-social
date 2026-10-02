
from genome import GenomePE
from run_episode import run_episode


NUM_AGENTS = 20


def reproduce(survivors):

    offspring_counts = [5, 5, 4, 3, 3]

    new_genomes = []

    for parent, count in zip(survivors, offspring_counts):

        for i in range(count):

            child = parent.clone()

            # Keep first offspring identical, mutate the rest
            if i > 0:
                child.mutate()

            new_genomes.append(child)

    return new_genomes


def evolve(generations=200):

    hall_of_fame = []
    genomes = [GenomePE() for _ in range(NUM_AGENTS)]

    for g in range(generations):

        fitness = run_episode(genomes)

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

    return genomes, hall_of_fame
