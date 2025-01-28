import sys
import time

import imageio
from tqdm import tqdm

sys.path.append("..")

import argparse

from flock.agents import MyEnvironment


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Initialize parameters for the boid simulation."
    )

    parser.add_argument(
        "--population", type=int, default=100, help="Population size (default: 10)"
    )
    parser.add_argument(
        "--cutoff_att",
        type=float,
        default=2.0,
        help="Attraction cutoff distance (default: 3.0)",
    )
    parser.add_argument(
        "--cutoff_rep",
        type=float,
        default=1.0,
        help="Repulsion cutoff distance (default: 1.0)",
    )
    parser.add_argument(
        "--cohesion_factor",
        type=float,
        default=0.1,
        help="Cohesion factor (default: 0.5)",
    )
    parser.add_argument(
        "--alignment_factor",
        type=float,
        default=0.2,
        help="Alignment factor (default: 0.2)",
    )
    parser.add_argument(
        "--separation_factor",
        type=float,
        default=5.0,
        help="Separation factor (default: 5.0)",
    )
    parser.add_argument(
        "--repulsion_factor",
        type=float,
        default=5.0,
        help="Repulsion factor (default: 1.0)",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=20.0,
        help="Half-dize of the box environment (default: 20.0)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=400,
        help="Maximum number of iterations (default: 400)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Delta time for every step (default: 0.04)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60,
        help="Frames per second for the final visualization (default: 60)",
    )
    parser.add_argument(
        "--show_forces",
        action="store_true",
        help="Show forces in the visualization? (default: False).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arguments()
    buffer = []
    blue_sky = MyEnvironment(
        population=args.population,
        cutoff_att=args.cutoff_att,
        cutoff_rep=args.cutoff_rep,
        cohesion_factor=args.cohesion_factor,
        alignment_factor=args.alignment_factor,
        separation_factor=args.separation_factor,
        repulsion_factor=args.repulsion_factor,
        box_size=args.box_size,
    )
    for _ in tqdm(range(args.max_steps)):

        blue_sky.step(dt=args.dt, save_forces=args.show_forces)
        frame = blue_sky.visualize(show_forces=args.show_forces)
        buffer.append(frame)

    # create the visualization gif
    with imageio.get_writer("../figures/output.gif", mode="I", fps=args.fps) as writer:
        for frame in buffer:
            # read the buffer as an image
            image = imageio.v2.imread(frame)

            # append the image to the GIF
            writer.append_data(image)
