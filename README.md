## MD_ND

Simple molecular dynamics (MD) program to run simulations with the classical Lennard-Jones (LJ) potential in any number of dimensions.

Derivatives of the Helmholtz energy (according to Lustig) are sampled during the simulation as well as the radial pair distribution function (RDF). Additionally, the chemical potential is calculated using Widom's insertion method.

The program is parallelized with OpenMP and utilizes Linked Cells/Cell Lists. It heavily depends on the compiler optimizations and, thus, uses `constexpr` extensively.

The software can only conduct simulations with single-site particles. The restriction to the potential parameters $\sigma=1$, $\epsilon/k_B=1$ and $m=1$ plays a minor role since results for values other than 1.0 can be obtained by a dimensional analysis (see script `convert_units.py`). When running with the full LJ potential, long-range corrections are utilized but only valid for up to five dimensions.


### build

The program requires C++26 or later due to the use of `constexpr` for cmath functions (e.g., `std::pow`). Furthermore, an OpenMP library (`libomp-dev`) must be installed.

Run make command to build

```bash
make
```


### run

Run simulation with

```bash
OMP_NUM_THREADS=8 ./MD_ND
```

You might set any other number of OpenMP threads.


### output

The simulation generates the following output files:
- `results_simsteps.dat`: Results printed during the simulation run. Averaged over simulation.
- `results_final.dat`: Final results of simulation. Averaged over whole production phase.
- `trajectory.vis`: Particle positions at given timesteps. Legacy format for visualizing trajectory.
- `RDF.dat`: Radial distribution function.

The simulation errors must be calculated in the post-processing, e.g., based on the `results_simsteps.dat` file.

Further thermodynamic properties including their errors can be determined using the script `calcThermoProperties.py`.

All quantities are given in reduced units (length parameter: 1 Angström; energy parameter: 1 K*k_B; mass parameter: 1 u). They can be converted to SI units using `convert_units.py`. The reference units might be changed to the potential parameters of , e.g., Argon if needed.


### validation

The simulation program was validated by comparing its results to the following literature data/software:

LJfull:
- 1D: [10.1063/1.458172](https://doi.org/10.1063/1.458172)
- 2D: [10.1139/p86-125](https://doi.org/10.1139/p86-125)
- 3D: [10.1016/j.cpc.2025.109541](https://doi.org/10.1016/j.cpc.2025.109541)
- 4D: [10.1063/1.480138](https://doi.org/10.1063/1.480138)

LJTS:
- 3D: [10.1016/j.cpc.2025.109541](https://doi.org/10.1016/j.cpc.2025.109541)

Furthermore, the calculated Helmholtz energy derivatives were compared to the value determined by the difference quotient for $D<6$.


### license

This software is based on [10.14279/depositonce-22395](https://doi.org/10.14279/depositonce-22395) and is subject to the [GNU General Public License 3.0 (GNU GPLv3)](https://choosealicense.com/licenses/gpl-3.0/). 

Upon usage, you agree to cite the following work: [10.1016/j.molliq.2025.127529](https://doi.org/10.1016/j.molliq.2025.127529)
