from itertools import product
from pathlib import Path

from ovito.io import import_file
from ovito.modifiers import SelectTypeModifier, DeleteSelectedModifier
from cowley_sro_parameters import nearest_neighbor_topology, sro_modifier
import matplotlib.pyplot as plt
import numpy as np


TEMPERATURES: list[int] = list(range(100, 500, 100))
METALS: list[str] = ["W", "Mo", "Ta"]
CHALCOGEN_PAIRS: list[tuple[str, str]] = [("S", "Te"), ("Se", "Te")]
MC_DATA_PATH: Path = Path("mc-data")


def main():

    fig, axs = plt.subplots(nrows=len(METALS), ncols=len(CHALCOGEN_PAIRS), sharex=True, sharey=True)

    for ax, (metal, chalcogen_pair) in zip(
        axs.ravel(),
        product(METALS, CHALCOGEN_PAIRS),
        strict=True
    ):

        print(metal, chalcogen_pair)

        average_sro = []
        std_sro = []

        for temperature in TEMPERATURES:

            print(temperature)

            pipeline = import_file(
                MC_DATA_PATH / f"{metal}{''.join(chalcogen_pair)}/{temperature:0d}/frame_*.xyz"
            )
            pipeline.modifiers.append(SelectTypeModifier(operate_on="particles", property="Particle Type", types={metal}))
            pipeline.modifiers.append(DeleteSelectedModifier())
            pipeline.modifiers.append(nearest_neighbor_topology(6))
            pipeline.modifiers.append(sro_modifier())

            sros = []
            for data in pipeline.frames:

                sros.append(data.attributes[f"sro_{''.join(chalcogen_pair)}"])
            
            average_sro.append(np.mean(sros[len(sros) // 2:]))
            std_sro.append(np.std(sros[len(sros) // 2:]))

        ax.grid()
        ax.set_ylabel(rf"$\alpha\beta=\text{{{''.join(chalcogen_pair)}}}$")
        ax.set_title(rf"$\text{{{metal}}}(\text{{{', '.join(chalcogen_pair)}}})_2$")
        ax.axhline(0, linestyle="--", color="black", label="fully random")
        ax.errorbar(TEMPERATURES, average_sro, yerr=std_sro, capsize=3, label=r"mean SRO $\pm$ $1\sigma$")

    axs[0, -1].legend()
    fig.supxlabel(r"temperature (K)")
    fig.supylabel(r"Cowley SRO parameter $\chi_{\alpha\beta}$")
    fig.tight_layout()
    fig.savefig("figures/mxx-ordering.pdf", dpi=800)


if __name__ == "__main__":

    main()
