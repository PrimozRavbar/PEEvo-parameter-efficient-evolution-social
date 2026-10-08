
import torch


R_IN  = 18    # 128x300
R_H   = 13    # 128x128
R_OUT = 2     # 10x128

NUM_PARAMETERS_PE = (
    128 * R_IN + R_IN * 300 +
    128 * R_H  + R_H * 128 +
    10 * R_OUT + R_OUT * 128
)

NUM_PARAMETERS_FULL = (
    128 * 300 +
    128 * 128 +
    10 * 128
)

NUM_HYPERPARAMETERS = 20


class GenomePE:

    def __init__(self, full_rank=False, initialization="random"):

        self.full_rank = full_rank

        if full_rank:

            if initialization == "zeros":

                weights = torch.zeros(NUM_PARAMETERS_FULL)

            elif initialization == "random":

                weights = torch.randn(NUM_PARAMETERS_FULL) * 0.1

            else:

                raise ValueError(
                    "initialization must be 'zeros' or 'random'"
                )

            self.chromosome = torch.cat([
                weights,
                torch.ones(NUM_HYPERPARAMETERS)
            ])

        else:

            Ain = torch.zeros(128, R_IN)
            Bin = torch.randn(R_IN, 300) * 0.1

            Ah = torch.zeros(128, R_H)
            Bh = torch.randn(R_H, 128) * 0.1

            Aout = torch.zeros(10, R_OUT)
            Bout = torch.randn(R_OUT, 128) * 0.1

            self.chromosome = torch.cat([
                Ain.flatten(),
                Bin.flatten(),
                Ah.flatten(),
                Bh.flatten(),
                Aout.flatten(),
                Bout.flatten(),
                torch.ones(NUM_HYPERPARAMETERS)
            ])

    def clone(self):

        g = GenomePE(full_rank=self.full_rank)
        g.chromosome = self.chromosome.clone()

        return g

    def mutate(self, mutation_rate=0.1, mutation_scale=0.05):

        if self.full_rank:

            weights = self.chromosome[:NUM_PARAMETERS_FULL]

            mask = (
                (weights != 0)
                & (torch.rand(NUM_PARAMETERS_FULL) < mutation_rate)
            )

            self.chromosome[:NUM_PARAMETERS_FULL] += (
                mask
                * torch.randn(NUM_PARAMETERS_FULL)
                * mutation_scale
            )

            hyperparameters = self.chromosome[NUM_PARAMETERS_FULL:]

            hyper_mask = (
                torch.rand(NUM_HYPERPARAMETERS)
                < mutation_rate
            )

            hyperparameters += (
                hyper_mask
                * torch.randn(NUM_HYPERPARAMETERS)
                * mutation_scale
            )

        else:

            mask = torch.rand_like(self.chromosome) < mutation_rate

            self.chromosome += (
                mask
                * torch.randn_like(self.chromosome)
                * mutation_scale
            )

    def add_remove_connection(
        self,
        mutation_rate=0.1,
        mutation_scale=0.05
    ):

        if not self.full_rank:
            return

        mask = torch.rand(NUM_PARAMETERS_FULL) < mutation_rate

        for i in torch.where(mask)[0]:

            if self.chromosome[i].item() == 0:

                self.chromosome[i] = (
                    torch.randn(1).item()
                    * mutation_scale
                )

            else:

                self.chromosome[i] = 0.0

    def extract_weights(self):

        if self.full_rank:

            idx = 0

            n_Win = 128 * 300
            n_Wh = 128 * 128
            n_Wout = 10 * 128

            Win = self.chromosome[
                idx:idx+n_Win
            ].reshape(128, 300)

            idx += n_Win

            Wh = self.chromosome[
                idx:idx+n_Wh
            ].reshape(128, 128)

            idx += n_Wh

            Wout = self.chromosome[
                idx:idx+n_Wout
            ].reshape(10, 128)

            return Win, Wh, Wout

        pe_chromosome = self.chromosome[:NUM_PARAMETERS_PE]

        idx = 0

        n_Ain = 128 * R_IN
        n_Bin = R_IN * 300
        n_Ah = 128 * R_H
        n_Bh = R_H * 128
        n_Aout = 10 * R_OUT
        n_Bout = R_OUT * 128

        Ain = pe_chromosome[
            idx:idx+n_Ain
        ].reshape(128, R_IN)

        idx += n_Ain

        Bin = pe_chromosome[
            idx:idx+n_Bin
        ].reshape(R_IN, 300)

        idx += n_Bin

        Ah = pe_chromosome[
            idx:idx+n_Ah
        ].reshape(128, R_H)

        idx += n_Ah

        Bh = pe_chromosome[
            idx:idx+n_Bh
        ].reshape(R_H, 128)

        idx += n_Bh

        Aout = pe_chromosome[
            idx:idx+n_Aout
        ].reshape(10, R_OUT)

        idx += n_Aout

        Bout = pe_chromosome[
            idx:idx+n_Bout
        ].reshape(R_OUT, 128)

        Win = Ain @ Bin
        Wh = Ah @ Bh
        Wout = Aout @ Bout

        return Win, Wh, Wout

    def extract_hyperparameters(self):

        offset = (
            NUM_PARAMETERS_FULL
            if self.full_rank
            else NUM_PARAMETERS_PE
        )

        return self.chromosome[
            offset:offset + NUM_HYPERPARAMETERS
        ]
