import torch


R_IN  = 18    # 128x300
R_H   = 13    # 128x128
R_OUT = 2     # 10x128

NUM_PARAMETERS_PE = (
    128 * R_IN + R_IN * 300 +       # Ain, Bin
    128 * R_H  + R_H * 128 +        # Ah, Bh
    10 * R_OUT + R_OUT * 128       # Aout, Bout
)


class GenomePE:

    def __init__(self):

        self.chromosome = torch.randn(NUM_PARAMETERS_PE) * 0.1

    def clone(self):

        g = GenomePE()
        g.chromosome = self.chromosome.clone()

        return g

    def mutate(self, mutation_rate=0.1, mutation_scale=0.05):

        mask = torch.rand_like(self.chromosome) < mutation_rate

        self.chromosome += (
            mask
            * torch.randn_like(self.chromosome)
            * mutation_scale
        )

        self.chromosome = torch.clamp(
            self.chromosome,
            -1.0,
            1.0
        )

    def extract_weights(self):

        chromosome = self.chromosome
        idx = 0

        n_Ain = 128 * R_IN
        n_Bin = R_IN * 300
        n_Ah = 128 * R_H
        n_Bh = R_H * 128
        n_Aout = 10 * R_OUT
        n_Bout = R_OUT * 128

        Ain = chromosome[idx:idx+n_Ain].reshape(128, R_IN)
        idx += n_Ain

        Bin = chromosome[idx:idx+n_Bin].reshape(R_IN, 300)
        idx += n_Bin

        Ah = chromosome[idx:idx+n_Ah].reshape(128, R_H)
        idx += n_Ah

        Bh = chromosome[idx:idx+n_Bh].reshape(R_H, 128)
        idx += n_Bh

        Aout = chromosome[idx:idx+n_Aout].reshape(10, R_OUT)
        idx += n_Aout

        Bout = chromosome[idx:idx+n_Bout].reshape(R_OUT, 128)

        Win = Ain @ Bin
        Wh = Ah @ Bh
        Wout = Aout @ Bout

        return Win, Wh, Wout
