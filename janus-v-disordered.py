from ase.build import mx2
from tce.calculator import TCECalculator
import numpy as np


def main():

    rng = np.random.default_rng(seed=0)
    calc = TCECalculator.load("tmd-tce.pkl")

    for metal in ["W", "Mo", "Ta"]:
        for chalcogen_pair in [("S", "Te"), ("Se", "Te")]:

            solution = mx2(f"{metal}{chalcogen_pair[0]}2", size=(12, 12, 1), vacuum=20.0)
            mask = solution.symbols == chalcogen_pair[0]
            solution.symbols[mask] = rng.choice(chalcogen_pair, size=sum(mask))

            janus = mx2(f"{metal}{''.join(chalcogen_pair)}", size=(12, 12, 1), vacuum=20.0)
            janus_energy = calc.get_potential_energy(janus) / len(janus)
            solution_energy = calc.get_potential_energy(solution) / len(solution)

            e_diff = janus_energy.item() - solution_energy.item()
            print(f"{metal}{''.join(chalcogen_pair)}: {e_diff=:.2f} eV/atom")


if __name__ == "__main__":

    main()
