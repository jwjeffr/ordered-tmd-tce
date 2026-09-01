from functools import wraps
from pathlib import Path
from itertools import product
import argparse
import logging
import sys

from ase.build import mx2
from ase.data import atomic_numbers
from ase import Atoms, io
import numpy as np
from tce.calculator import TCECalculator
from tce.monte_carlo import monte_carlo, MCStep
from pathos.multiprocessing import ProcessPool


RNG: np.random.Generator = np.random.default_rng(seed=0)
METALS: list[str] = ["W", "Mo", "Ta"]
CHALCOGEN_PAIRS: list[tuple[str]] = [("S", "Te"), ("Se", "Te")]
BOLTZMANN_CONSTANT_EV_PER_K: float = 8.617e-5
MC_DATA_DIR: Path = Path("mc-data")


def sublattice_swap_factory(
    swappable_atomic_numbers: np.typing.NDArray[np.int64]
) -> MCStep:

    if not isinstance(swappable_atomic_numbers, np.ndarray):

        swappable_atomic_numbers = np.array(swappable_atomic_numbers)

    @wraps(sublattice_swap_factory)
    def wrapper(atoms: Atoms) -> Atoms:

        new_atoms = atoms.copy()

        mask = new_atoms.numbers[:, np.newaxis] == swappable_atomic_numbers[np.newaxis, :]
        swappable_sites, = np.where(np.any(mask, axis=1))

        i, j = RNG.choice(swappable_sites, size=2, replace=False)
        while new_atoms.numbers[i] == new_atoms.numbers[j]:
            i, j = RNG.choice(swappable_sites, size=2, replace=False)

        new_atoms.numbers[i], new_atoms.numbers[j] = new_atoms.numbers[j], new_atoms.numbers[i]
        return new_atoms

    return wrapper


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("temperature", type=int)
    args = parser.parse_args()
    beta = 1.0 / (BOLTZMANN_CONSTANT_EV_PER_K * args.temperature)

    chalcogen_atomic_numbers = list(set(sum(
        [[atomic_numbers[t1], atomic_numbers[t2]] for t1, t2 in CHALCOGEN_PAIRS],
        start=[]
    )))

    mc_step = sublattice_swap_factory(
        swappable_atomic_numbers=chalcogen_atomic_numbers
    )
    calc = TCECalculator.load("tmd-tce.pkl")

    def run_mc(metal: str, chalcogen_pair: tuple[str, str]) -> None:

        temperature_dir = MC_DATA_DIR / f"{metal}{''.join(chalcogen_pair)}" / f"{args.temperature:0d}"
        temperature_dir.mkdir(exist_ok=True, parents=True)

        slab = mx2(f"{metal}{chalcogen_pair[0]}2", size=(15, 15, 1), vacuum=20.0)
        mask = slab.symbols == chalcogen_pair[0]

        slab.symbols[mask] = RNG.choice(chalcogen_pair, size=sum(mask))

        trajectory = monte_carlo(
            initial_configuration=slab,
            tce_calculator=calc,
            num_steps=10_000,
            beta=beta,
            save_every=100,
            return_generator=True,
            mc_step=mc_step,
            callback=lambda x, y: None
        )

        for i, frame in enumerate(trajectory, start=1):

            print(f"Step {i:0d}: Energy = {frame.info['energy']:.2f} eV")
            io.write(temperature_dir / f"frame_{i:0d}.xyz", frame, format="extxyz")

    with ProcessPool(nodes=6) as pool:

        pool.map(lambda x: run_mc(*x), product(METALS, CHALCOGEN_PAIRS))


if __name__ == "__main__":

    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    main()
