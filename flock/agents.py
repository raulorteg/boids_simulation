import io

import matplotlib.pyplot as plt
import numpy as np
import torch


class MyEnvironment(object):
    def __init__(
        self,
        population: int = 10,
        ndim: int = 2,
        cutoff_att: float = 1.0,
        cutoff_rep: float = 0.5,
        cohesion_factor: float = 0.2,
        alignment_factor: float = 0.2,
        separation_factor: float = 0.2,
        repulsion_factor: float = 3.0,
        box_size: float = 20.0,
    ):
        self.population = population
        self.ndim = ndim
        self.box_size = box_size

        self.cutoff_att = cutoff_att
        self.cutoff_rep = cutoff_rep

        self.cohesion_factor = cohesion_factor
        self.alignment_factor = alignment_factor
        self.separation_factor = separation_factor
        self.repulsion_factor = repulsion_factor

        self.horizontal_mask = torch.zeros((self.population, self.ndim))
        self.horizontal_mask[:, 0] = 1

        self.vertical_mask = torch.zeros((self.population, self.ndim))
        self.vertical_mask[:, 1] = 1

        self.positions = (
            0.9 * self.box_size * torch.randn(size=(self.population, self.ndim))
        )

        self.self_perception_mask = torch.ones(
            (self.population, self.population)
        ) - torch.eye(self.population)

        self.speeds = 5.0 * torch.randn(size=(self.population, self.ndim)) + 1.0
        self.directions = torch.nn.functional.normalize(self.speeds, p=2.0, dim=1) / 5
        self.pairwise_distances = torch.cdist(self.positions, self.positions, p=2.0)

    def cohesion(self, cutoff: float):
        """
        Method that return forces pulling every boid into the center of its neighborhood.
        """

        # use the pairwise distance to produce a mask of local neighbors and avoid self-perception
        mask = (self.pairwise_distances <= cutoff) * self.self_perception_mask.bool()
        masked_sum = mask.sum(dim=1, keepdim=True)

        # compute the mean value across the neighbours (1e-6 to avoid a division by zero when no neighbors)
        mean = torch.matmul(mask.float(), self.positions) / (1e-6 + masked_sum)

        # for boids with no neighbors the mean value is their value
        mean[masked_sum.squeeze(1) == 0] = self.positions[masked_sum.squeeze(1) == 0]

        forces = self.cohesion_factor * (mean - self.positions)
        return forces

    def alignment(self, cutoff: float):
        """
        Method that return forces pulling every boid into the same direction as its neighborhood.
        """

        # use the pairwise distance to produce a mask of local neighbors and avoid self-perception
        mask = (self.pairwise_distances <= cutoff) * self.self_perception_mask.bool()
        masked_sum = mask.sum(dim=1, keepdim=True)

        # compute the mean value across the neighbours (1e-6 to avoid a division by zero when no neighbors)
        mean = torch.matmul(mask.float(), self.speeds) / (1e-6 + masked_sum)
        mean[masked_sum.squeeze(1) == 0] = self.speeds[masked_sum.squeeze(1) == 0]

        # for boids with no neighbors the mean value is their value
        forces = mean - self.speeds
        return self.alignment_factor * forces

    def separation(self, cutoff: float):
        """
        Method that return forces pushing away every boid from the center of its neighborhood.
        """

        # use the pairwise distance to produce a mask of local neighbors and avoid self-perception
        mask = (self.pairwise_distances <= cutoff) * self.self_perception_mask.bool()
        masked_sum = mask.sum(dim=1, keepdim=True)

        # compute the mean value across the neighbours (1e-6 to avoid a division by zero when no neighbors)
        mean = torch.matmul(mask.float(), self.positions) / (1e-6 + masked_sum)

        # for boids with no neighbors the mean value is their value
        mean[masked_sum.squeeze(1) == 0] = self.positions[masked_sum.squeeze(1) == 0]

        forces = -self.separation_factor * (mean - self.positions)
        return forces

    def repulsion(self):
        """
        Forces enforced near the box bounds to prevent the flock from escaping
        """

        left_repulsion = (
            self.positions < -0.8 * self.box_size
        ).int() * self.horizontal_mask

        rigt_repulsion = (
            -(self.positions > 0.8 * self.box_size).int() * self.horizontal_mask
        )

        lower_repulsion = (
            self.positions < -0.8 * self.box_size
        ).int() * self.vertical_mask

        upper_repulsion = (
            -(self.positions > 0.8 * self.box_size).int() * self.vertical_mask
        )

        return self.repulsion_factor * (
            left_repulsion + rigt_repulsion + lower_repulsion + upper_repulsion
        )

    def step(self, dt: float = 1.0, save_forces: bool = False):

        forces_cohesion = self.cohesion(cutoff=self.cutoff_att)
        forces_alignment = self.alignment(cutoff=self.cutoff_att)
        forces_separation = self.separation(cutoff=self.cutoff_rep)
        forces_repulsion = self.repulsion()

        total_forces = (
            forces_cohesion + forces_alignment + forces_separation + forces_repulsion
        )

        # Note: only used for visualizing forces
        if save_forces:
            self.force_plot = total_forces

        self.speeds = self.speeds + total_forces * dt
        self.speeds = (
            1.001 * self.speeds
        )  # flap a bit the wings, otherwise they slow down
        self.positions = (
            self.positions + self.speeds * dt + 0.5 * total_forces * (dt**2)
        )

        self.directions = (
            torch.nn.functional.normalize(self.speeds, p=2.0, dim=1) / 5
        )  # Note: scaling only used for better visualization
        self.pairwise_distances = torch.cdist(self.positions, self.positions, p=2.0)

    def visualize(self, show_forces: bool = False):

        plt.figure(figsize=(4, 4), facecolor="lightblue")
        plt.xlim(-self.box_size, self.box_size)
        plt.ylim(-self.box_size, self.box_size)

        for i in range(self.population):
            plt.arrow(
                self.positions[i, 0],
                self.positions[i, 1],
                self.directions[i, 0],
                self.directions[i, 1],
                width=0.2,
            )

        if show_forces:
            for i in range(self.population):
                plt.arrow(
                    self.positions[i, 0],
                    self.positions[i, 1],
                    self.force_plot[i, 0],
                    self.force_plot[i, 1],
                    width=0.1,
                    color="red",
                )

        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer.getvalue()
