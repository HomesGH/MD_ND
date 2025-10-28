
/*    Copyright (c) 2025
 *    This file is under license: GNU General Public License 3.0 (GNU GPLv3)
 *
 *    This software is based on https://doi.org/10.14279/depositonce-22395
 * 
 *    Upon usage, you agree to cite the following work:
 *    https://doi.org/10.1016/j.molliq.2025.127529
 */

#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#define LJTS false  // false -> LJfull; LJTS: Truncated and shifted Lennard-Jones potential

#define DEBUG_OUT false  // Debug output


constexpr short nDims = 3;  // Dimension

// State variables
constexpr double temperature = 2.0;  // Reduced temperature
constexpr double density = 0.1;      // Reduced density

// Cutoff limit
constexpr double r_cutoff = 4.0;  // Radius of the cutoff sphere

// Timestep parameters
constexpr int steps_equi = 50000;     // Number of equilibration timesteps
constexpr int steps_prod = 200000;    // Number of production timesteps
constexpr double delta_time = 0.003;  // Timestep width

// Control parameters
constexpr int num_prtls_dim = 10;  // Number of particles in one dimension
constexpr int num_prtls = std::pow(num_prtls_dim, nDims);  // Number of particles in the simulation

constexpr int num_prtls_chemPot = num_prtls;  // Number of test particles for sampling of chem. pot.

constexpr bool flg_ensemble_NVT = true;   // For NVT ensemble: flg_ensemble_NVT=true; otherwise NVE ensemble
constexpr bool flg_equi = true;           // For equilibration phase: flg_equi=true
constexpr bool flg_chemPot = true;        // For sampling the chem. pot.: flg_chemPot=true
constexpr int writefreq_output = 1000;    // Output of state variables after every writefreq_output timesteps
constexpr int writefreq_vis_RDF = 10000;  // Output for visualization and RDF after every writefreq_vis_RDF timesteps
constexpr double binwidth_RDF = 0.01;     // Binwidth for calculation of RDF

constexpr uint num_shells_RDF = r_cutoff / binwidth_RDF;  // Number of spherical shells in RDF calculation; RDF up to cutoff radius

// File names
const std::string filename_result = "results_simsteps.dat";  // Name of the file with results every writefreq_output timesteps
const std::string filename_final = "results_final.dat";      // Name of the file with final results
const std::string filename_vis = "trajectory.vis";           // Name of the visualization file
const std::string filename_RDF = "RDF.dat";                  // Name of the file with the radial distribution function (RDF)

// File header string
constexpr short file_columnwidth = 10;
inline std::string make_result_fileheader() {
    std::vector<std::string> headers = {"simstep", "temperature", "density", "pressure", "dUdV", "epot", "ekin", "etotal", "mu_res", "numTestMu",
                                        "A00r",    "A10r",        "A01r",    "A20r",     "A11r", "A02r", "A30r", "A21r",   "A12r"};
    std::ostringstream oss;
    for (const auto &h : headers) {
        oss << std::setw(file_columnwidth) << std::right << h << " ";
    }
    return oss.str();
}
const std::string result_fileheader = make_result_fileheader();

// Miscellaneous
constexpr double boxlength = std::pow((num_prtls / density), (1. / nDims));  // Edge length of the simulation volume
constexpr double boxlength_sqrt = boxlength * boxlength;
constexpr double volume = std::pow(boxlength, nDims);
constexpr double r_cutoff_sqrt = r_cutoff * r_cutoff;

#if (LJTS)
constexpr double upot_shifted = std::pow((1. / r_cutoff), 12) - std::pow((1. / r_cutoff), 6);  // 0.004079223 for rc=2.5
#else
constexpr double upot_shifted = 0.0;
#endif

// Sampling
// Bulk: Averaged over whole simulation
double U_LRC = 0.0;       // Long-range correction of potential energy
double p_LRC = 0.0;       // Long-range correction of pressure
double dUdV_LRC = 0.0;    // Long-range correction dUdV_LRC
double d2UdV2_LRC = 0.0;  // Long-range correction d2UdV2_LRC

double U_accum = 0.0;      // Accumulated (over timesteps) potential energy of all particles
double p_accum = 0.0;      // Accumulated pressure
double ekin_accum = 0.0;   // Accumulated kinetic energy
double U_step = 0.0;       // Potential energy in current simstep
double virial_step = 0.0;  // Virial in current simstep
// Lustig formalism
double dUdV = 0.0;        // dUdV
double d2UdV2 = 0.0;      // d2UdV2
double dUdV_accum = 0.0;  // Accumulated dUdV
double d2UdV2_accum = 0.0;  // Accumulated d2UdV2
double U2_accum = 0.0;
double U3_accum = 0.0;
double dUdV_2_accum = 0.0;
double U_dUdV_accum = 0.0;
double U_2_dUdV_accum = 0.0;
double U_dUdV_2_accum = 0.0;
double U_d2UdV_2_accum = 0.0;
// Chem. pot. sampling
double mu_accum = 0.0;  // Accumulated chemical potential
double mu_step = 0.0;   // Chemical potential in current simstep

unsigned long long num_test_accum = 0;  // Accumulated number of actual test particles for chem. pot. sampling
int num_test_step = 0;                  // Number of actual test particles in current simstep

unsigned long long count_RDF[num_shells_RDF] = {0};  // Vector of number of particles in each shell for RDF

// Vectors
std::vector<std::array<double, nDims>> prtl_positions;    // Position
std::vector<std::array<double, nDims>> prtl_velocities;   // Velocity
std::vector<std::array<double, nDims>> prtl_forces;       // Force
std::vector<std::array<double, nDims>> prtl_forces_prev;  // Force in the previous timestep

std::vector<int> cell_index;

// For grid of cells
constexpr int grid_n_part = std::floor(boxlength / r_cutoff);  // Number of cells in one direction
constexpr double grid_ddims = 1.0 / grid_n_part;               // Normalized width of one cell; Assuming scaled cubic boxlength of 1.0
constexpr int numCellsDirect1D = std::min(grid_n_part, 3);     // Number of cells to traverse in one direction;
constexpr int ncells = std::pow(grid_n_part, nDims);

// Each element(=cell) contains a vector that indicates which particles are in the respective cell
std::vector<std::vector<int>> cell_list;

// Own function for speed-optimized power calculations
template <typename T, typename T2>
requires std::integral<T2> constexpr T power(T base, T2 exponent) {
    if (exponent == 0) {
        return static_cast<T>(1);
    } else if (exponent < 0) {
        return static_cast<T>(1) / power(base, -exponent);
    } else {
        T result = base;
        for (T2 i = 1; i < exponent; ++i) {
            result *= base;
        }
        return result;
    }
}

// Own function for faster implementation of rounding function
constexpr double fastRound(const double x) {
    return (x >= 0.0) ? static_cast<int>(x + 0.5) : static_cast<int>(x - 0.5);
}

// Calculate factorial using recursion
constexpr unsigned long long factorial(int n) {
    return n > 1 ? n * factorial(n - 1) : 1;
}

// Generates a random number between 0.0 and 1.0
std::mt19937 rng;
double random_num(std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// Functions for linked cell algorithm
void cell_list_clear() {
    for (int i = 0; i < ncells; ++i) {
        cell_list[i].clear();
    }
}
void cell_list_compute_c() {
    for (int i = 0; i < num_prtls; ++i) {
        int i_dim[nDims];
        for (short d = 0; d < nDims; ++d) {
            i_dim[d] = prtl_positions[i][d] / grid_ddims;
        }

        int ic = 0;
        for (short d = 0; d < nDims; ++d) {
            int grid_index = 1;
            for (short j = d + 1; j < nDims; ++j) {
                grid_index *= grid_n_part;
            }
            ic += i_dim[d] * grid_index;
        }

        cell_index[i] = ic;

        if (ic < 0) {
            std::cerr << "ERROR: Compute cell list: bad_ic (too small) " << ic << " < 0 for particle at ";
            for (short d = 0; d < nDims; ++d) {
                std::cerr << prtl_positions[i][d] << " ";
            }
            std::cerr << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (ic >= ncells) {
            std::cerr << "ERROR: Compute cell list: bad_ic (too large) " << ic << " >= " << ncells << " for particle at ";
            for (short d = 0; d < nDims; ++d) {
                std::cerr << prtl_positions[i][d] << " ";
            }
            std::cerr << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}
void cell_list_build() {
    for (int i = 0; i < num_prtls; ++i) {
        cell_list[cell_index[i]].push_back(i);
    }
}

// Calculate the gamma function of nDims/2
double gamma_factor() {
    double gamma;
    if (nDims % 2 == 0) {
        gamma = factorial((nDims / 2) - 1);
    } else {
        int k_factor = (nDims - 1) / 2;
        gamma = (factorial(nDims - 1) / (factorial(k_factor) * std::pow(4, k_factor))) * std::sqrt(std::numbers::pi);
    }
    return gamma;
}

// Output for visualization
void writeVis() {
    std::ofstream visFile;
    visFile.open(filename_vis, std::ios::app);
    visFile << "# " << boxlength << " new Frame" << std::endl;
    for (int i = 0; i < num_prtls; ++i) {
        if (nDims == 1) {
            visFile << "! 1"
                    << " " << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][0] << " 0"
                    << " 0"
                    << " 1 998 0 0" << std::endl;
        } else if (nDims == 2) {
            visFile << "! 1"
                    << " " << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][0] << " " << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][1] << " 0"
                    << " 1 998 0 0" << std::endl;
        } else {
            visFile << "! 1"
                    << " " << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][0] << " " << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][1] << " "
                    << std::setw(12) << std::setprecision(10) << 1000 * prtl_positions[i][2] << " 1 998 0 0" << std::endl;
        }
    }
    visFile << "\n" << std::endl;
    visFile.close();
}

// Calculation of forces and virial using the 12-6 LJ potential
void calcPotForceLJ() {
    const double r_cutoff_sqrt_scaled = r_cutoff_sqrt / boxlength_sqrt;
    const double binwidth_RDF_scaled = r_cutoff / (boxlength * num_shells_RDF);
    const int numCellsDirect = power(numCellsDirect1D, nDims);  // Number of direct neighbor cells

    U_step = 0.;
    virial_step = 0.;
    dUdV = 0.;
    d2UdV2 = 0.;

    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            prtl_forces_prev[i][d] = prtl_forces[i][d];
            prtl_forces[i][d] = 0;
        }
    }

#pragma omp parallel default (none) \
    shared(std::cout,std::cerr,num_prtls,boxlength,boxlength_sqrt,ncells,cell_index,cell_list,prtl_positions,prtl_forces) \
    shared(numCellsDirect,numCellsDirect1D,r_cutoff_sqrt_scaled,binwidth_RDF_scaled,upot_shifted) \
    reduction(+ : U_step,virial_step,d2UdV2,count_RDF[:num_shells_RDF])
    {
        U_step = 0.;
        virial_step = 0.;
        d2UdV2 = 0.;

#pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < num_prtls; ++i) {
            int ic = cell_index[i];
            int i_dim[nDims];
            // std::vector<int> processedCells;

            for (short d = 0; d < nDims; ++d) {
                int ic_temp = ic;
                int grid_index_denom = 1;
                for (short j = nDims - 1; j > d; --j) {
                    grid_index_denom *= grid_n_part;
                }
                for (short d2 = 0; d2 < d; ++d2) {
                    int grid_index_subtr = i_dim[d2];
                    for (short k = nDims - 1; k > d2; --k) {
                        grid_index_subtr *= grid_n_part;
                    }
                    ic_temp -= grid_index_subtr;
                }
                i_dim[d] = ic_temp / grid_index_denom;
            }

            // Loop over surrounding grid cells
            for (int c_num = 0; c_num < numCellsDirect; ++c_num) {
                int c[nDims];           // value -1, 0 or +1
                int i_surr_dim[nDims];  // Dimensionwise index of surrounding grid cell

                for (short d = 0; d < nDims; ++d) {
                    c[d] = ((static_cast<int>(c_num / power(numCellsDirect1D, d))) % numCellsDirect1D) - 1;  // e.g. =MOD(FLOOR(c_num/9),3) ; value -1, 0 or +1
                }

                // Check if at boundary --> PBC
                for (short d = 0; d < nDims; ++d) {
                    i_surr_dim[d] = i_dim[d] + c[d];
                    if (i_surr_dim[d] < 0) {
                        i_surr_dim[d] += grid_n_part;
                    }
                    if (i_surr_dim[d] > grid_n_part - 1) {
                        i_surr_dim[d] -= grid_n_part;
                    }
                }

                // Calculate index of surrounding grid cell
                int ic_surr = 0;  // Index of surrounding grid cell
                for (short d = 0; d < nDims; ++d) {
                    int grid_index = 1;
                    for (short j = d + 1; j < nDims; ++j) {
                        grid_index *= grid_n_part;
                    }
                    ic_surr += i_surr_dim[d] * grid_index;
                }

                for (size_t k = 0; k < cell_list[ic_surr].size(); ++k) {  // k: Index of particle within cell_list
                    int j = cell_list[ic_surr][k];                        // j: Global index of particle
                    if (i != j) {
                        double distance[nDims];
                        double distance_sqrt = 0.0;  // without boxlength

                        for (short d = 0; d < nDims; ++d) {
                            distance[d] = prtl_positions[i][d] - prtl_positions[j][d];
                            distance[d] = distance[d] - fastRound(distance[d]);  // Minimum Image Convention
                            distance_sqrt += distance[d] * distance[d];
                        }

                        // Only proceed if within the cutoff radius
                        if (distance_sqrt <= r_cutoff_sqrt_scaled) {
                            const double distance_sqrt_inv_scaled = 1. / (distance_sqrt * boxlength_sqrt);
                            const double dist_r6_inv_scaled = distance_sqrt_inv_scaled * distance_sqrt_inv_scaled * distance_sqrt_inv_scaled;
                            const double dist_r12_inv_scaled = dist_r6_inv_scaled * dist_r6_inv_scaled;
                            const double force_ij = 24. * (2. * dist_r12_inv_scaled - dist_r6_inv_scaled) * distance_sqrt_inv_scaled * boxlength;

                            for (short d = 0; d < nDims; ++d) {
                                prtl_forces[i][d] += force_ij * distance[d];
                            }
                            U_step += 0.5 * (dist_r12_inv_scaled - dist_r6_inv_scaled + upot_shifted);   // half energy
                            virial_step += 0.5 * 24. * (2. * dist_r12_inv_scaled - dist_r6_inv_scaled);  // half virial
                            // For Lustig formalism
                            d2UdV2 += 0.5 * ((2 * (nDims - 1) + 26) * dist_r12_inv_scaled - (((nDims - 1) + 7) * dist_r6_inv_scaled));  // half d2UdV2
                            // RDF
                            const uint indexRDF = uint(std::sqrt(distance_sqrt) / binwidth_RDF_scaled);
                            count_RDF[indexRDF] += 1;
                        }
                    }
                }
            }
        }
    }

    U_step *= 4.;
    virial_step *= (1. / nDims);  // From here on virial_step is approximately the residual pressure due to factor 1./nDims
    dUdV = -virial_step * density / num_prtls;
    d2UdV2 *= 24. / (nDims * nDims * volume * volume);
}

// Calculation of chemical potential
void calcChemicalPotential() {
    const double r_cutoff_sqrt_scaled = r_cutoff_sqrt / boxlength_sqrt;
    const int numCellsDirect = power(numCellsDirect1D, nDims);  // Number of direct neighbor cells

    mu_step = 0.;
    num_test_step = 0;

#pragma omp parallel default (none) \
    shared (std::cout,std::cerr,num_prtls,boxlength,boxlength_sqrt,ncells,cell_index,cell_list,prtl_positions) \
    shared(rng,grid_ddims,numCellsDirect,numCellsDirect1D,r_cutoff_sqrt_scaled,upot_shifted,U_LRC) \
    reduction(+ : mu_step,num_test_step)
    {
        mu_step = 0.;
        num_test_step = 0;

#pragma omp for schedule(dynamic, 4)
        for (int i_test = 0; i_test < num_prtls_chemPot; ++i_test) {
            double Q0_testPrtl[nDims];
            double dU_testPrtl = 0.0;

            // Random position of test particle
            for (short d = 0; d < nDims; ++d) {
                Q0_testPrtl[d] = 0.98 * random_num(rng) + 0.01;  // Position between 0.99 and 0.01 to prevent errors at boundaries
            }

            // Get index of cell where test particle would be
            int i_dim_testPrtl[nDims];
            for (short d = 0; d < nDims; ++d) {
                i_dim_testPrtl[d] = Q0_testPrtl[d] / grid_ddims;
            }
            int ic_testPrtl = 0;  // Index of cell where test particle would be
            for (short d = 0; d < nDims; ++d) {
                int grid_index = 1;
                for (short j = d + 1; j < nDims; ++j) {
                    grid_index *= grid_n_part;
                }
                ic_testPrtl += i_dim_testPrtl[d] * grid_index;
            }
#if (DEBUG_OUT)
            std::cout << "Test particle " << i_test << " at ";
            for (short d = 0; d < nDims; ++d) {
                std::cout << Q0_testPrtl[d] << " ";
            }
            std::cout << "; index: " << ic_testPrtl << std::endl;
#endif
            if (ic_testPrtl < 0) {
                std::cerr << "ERROR: ChemPot sampling: bad_ic (too small) " << ic_testPrtl << " < 0 for particle at ";
                for (short d = 0; d < nDims; ++d) {
                    std::cerr << Q0_testPrtl[d] << " ";
                }
                std::cerr << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (ic_testPrtl >= ncells) {
                std::cerr << "ERROR: ChemPot sampling: bad_ic (too large) " << ic_testPrtl << " >= " << ncells << " for particle at ";
                for (short d = 0; d < nDims; ++d) {
                    std::cerr << Q0_testPrtl[d] << " ";
                }
                std::cerr << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Loop over surrounding grid cells
            for (int c_num = 0; c_num < numCellsDirect; ++c_num) {
                int c[nDims];                    // value -1, 0 or +1
                int i_surr_dim_testPrtl[nDims];  // Dimensionwise index of surrounding grid cell

                for (short d = 0; d < nDims; ++d) {
                    c[d] = ((static_cast<int>(c_num / power(numCellsDirect1D, d))) % numCellsDirect1D) - 1;  // e.g. =MOD(FLOOR(c_num/9),3) ; value -1, 0 or +1
                }

                // Check if at boundary --> PBC
                for (short d = 0; d < nDims; ++d) {
                    i_surr_dim_testPrtl[d] = i_dim_testPrtl[d] + c[d];
                    if (i_surr_dim_testPrtl[d] < 0) {
                        i_surr_dim_testPrtl[d] += grid_n_part;
                    }
                    if (i_surr_dim_testPrtl[d] > grid_n_part - 1) {
                        i_surr_dim_testPrtl[d] -= grid_n_part;
                    }
                }

                // Calculate index of surrounding grid cell
                int ic_surr_testPrtl = 0;  // Index of surrounding grid cell
                for (short d = 0; d < nDims; ++d) {
                    int grid_index = 1;
                    for (short j = d + 1; j < nDims; ++j) {
                        grid_index *= grid_n_part;
                    }
                    ic_surr_testPrtl += i_surr_dim_testPrtl[d] * grid_index;
                }
#if (DEBUG_OUT)
                std::cout << "Index of surrounding grid cell " << ic_surr_testPrtl << std::endl;
#endif

                for (size_t k = 0; k < cell_list[ic_surr_testPrtl].size(); ++k) {  // k: Index of particle within cell_list
                    int j = cell_list[ic_surr_testPrtl][k];                        // j: Global index of particle

                    double distance[nDims];
                    double distance_sqrt = 0.0;

                    for (short d = 0; d < nDims; ++d) {
                        distance[d] = Q0_testPrtl[d] - prtl_positions[j][d];
                        distance[d] = distance[d] - fastRound(distance[d]);  // Minimum Image Convention
                        distance_sqrt += distance[d] * distance[d];
                    }

                    // Only proceed if within the cutoff radius
                    if (distance_sqrt <= r_cutoff_sqrt_scaled) {
                        const double distance_sqrt_inv_scaled = 1. / (distance_sqrt * boxlength_sqrt);
                        const double dist_r6_inv_scaled = distance_sqrt_inv_scaled * distance_sqrt_inv_scaled * distance_sqrt_inv_scaled;
                        const double dist_r12_inv_scaled = dist_r6_inv_scaled * dist_r6_inv_scaled;

                        const double energy_insertion = dist_r12_inv_scaled - dist_r6_inv_scaled + upot_shifted;
                        if (std::isfinite(energy_insertion)) {
                            dU_testPrtl += energy_insertion;
                        }
#if (DEBUG_OUT)
                        std::cout << "Test particle " << i_test << " (cell " << ic_testPrtl << ") interacting with particle " << j << " (cell " << ic_surr_testPrtl << ") at ";
                        for (short d = 0; d < nDims; ++d) {
                            std::cout << prtl_positions[j][d] << " ";
                        }
                        std::cout << "distance distance_sqrt: " << distance_sqrt << " dU " << 0.5 * (4. * (dist_r12_inv_scaled - dist_r6_inv_scaled + upot_shifted))
                                  << " dist_r12_inv_scaled: " << dist_r12_inv_scaled << " dist_r6_inv_scaled " << dist_r6_inv_scaled << std::endl;
#endif
                    }
                }
            }
            dU_testPrtl *= 4.;
            dU_testPrtl += 2. * density * U_LRC;
            double chemPot = std::exp(-dU_testPrtl / temperature);
            if (std::isfinite(chemPot)) {
                mu_step += chemPot;
                num_test_step++;
#if (DEBUG_OUT)
                std::cout << "Test particle " << i_test << " (cell " << ic_testPrtl << ") has chemical potential of " << chemPot << " due to dU = " << dU_testPrtl << std::endl;
#endif
            }
        }
    }
}

// Calculation of the long-range corrections for potential energy (derivatives) and pressure
void calcCutoffCorrections() {
#if (LJTS)
    U_LRC = 0.0;
    p_LRC = 0.0;
    dUdV_LRC = 0.0;
    d2UdV2_LRC = 0.0;
#else
    // LRC is only valid up to 5D
    if (nDims > 5) {
        std::cerr << "ERROR: LRC does not support dimensionality greater than 5";
        std::exit(EXIT_FAILURE);
    }
    double gamma_pi_factor = 4. * (std::pow(std::numbers::pi, 0.5 * nDims) / gamma_factor());

    // Corrections without density (as in ms2)
    U_LRC = -gamma_pi_factor * ((-1. / ((12 - nDims) * std::pow(r_cutoff, 12 - nDims))) + (1. / ((6 - nDims) * std::pow(r_cutoff, 6 - nDims))));
    dUdV_LRC = (1. / (nDims)) * gamma_pi_factor * ((12. / ((12 - nDims) * std::pow(r_cutoff, 12 - nDims))) - (6. / ((6 - nDims) * std::pow(r_cutoff, 6 - nDims))));

    p_LRC = dUdV_LRC;

    double d2UdV2_CORR_2 = (1. / (nDims * nDims)) * gamma_pi_factor * (((-13. * 12.) / ((12 - nDims) * std::pow(r_cutoff, 12 - nDims))) + ((6. * 7.) / ((6 - nDims) * std::pow(r_cutoff, 6 - nDims))));
    d2UdV2_LRC = -d2UdV2_CORR_2;

    std::cout << "U_LRC      = " << U_LRC << std::endl;
    std::cout << "p_LRC      = " << p_LRC << std::endl;
    std::cout << "dUdV_LRC   = " << dUdV_LRC << std::endl;
    std::cout << "d2UdV2_LRC = " << d2UdV2_LRC << std::endl;

#endif
}

// Setup of the initial grid
void makeLattice() {
    float num_prtls_dim_temp = std::pow(num_prtls, (1. / (static_cast<float>(nDims))));  // Number of particles per dimension
    float dist_dim = 1. / num_prtls_dim_temp;                                            // Distance between two particles in one direction

    if ((static_cast<int>(num_prtls_dim_temp) - num_prtls_dim_temp) != 0) {
        std::cerr << "ERROR: Please choose one of the following number of particles (k^nDims with k being an integer): ";
        for (int i = 3; i < 12; ++i) {
            std::cerr << std::pow(i, nDims) << " ";
        }
        std::cerr << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int N_dim = static_cast<int>(num_prtls_dim_temp);  // Number of particles per dimension

    double shiftLattice = 0.0;
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            // Setup triangular lattice in case of 2D
            // shift every second row
            if (nDims == 2) {
                if (((i / N_dim) % 2 == 0) && (d == 0)) {
                    shiftLattice = 0.0;
                } else {
                    shiftLattice = 0.499 * dist_dim;
                }
            }
            prtl_positions[i][d] = ((static_cast<int>(i / std::pow(N_dim, d))) % N_dim) * dist_dim + shiftLattice;  // e.g. =MOD(FLOOR(i/25),5)
        }
    }
}

// Create output files
void initFiles() {
    std::ofstream resultFile;
    std::ofstream visFile;

    time_t now = std::time(0);

    resultFile.open(filename_result, std::ios::out);
    if (flg_ensemble_NVT) {
        resultFile << "Ensemble: NVT" << std::endl;
    } else {
        resultFile << "Ensemble: NVE" << std::endl;
    }
    resultFile << "Start of Simulation: " << std::ctime(&now) << std::endl;
    resultFile << "Dimension:     " << nDims << std::endl;
#if (LJTS)
    resultFile << "Fluid:         LJTS" << std::endl;
#else
    resultFile << "Fluid:         LJfull" << std::endl;
#endif
    resultFile << "NumPrtls:  " << num_prtls << std::endl;
    resultFile << "Boxlength: " << boxlength << std::endl;
    resultFile << "TimestepWidth:   " << delta_time << std::endl;
    resultFile << "CutoffRadius: " << std::fixed << std::setprecision(3) << r_cutoff << std::endl;
    resultFile << "U_LRC:        " << std::fixed << std::setprecision(6) << U_LRC << std::endl;
    resultFile << "p_LRC:        " << std::fixed << std::setprecision(6) << p_LRC << std::endl;
    resultFile << "dUdV_LRC:     " << std::fixed << std::setprecision(6) << dUdV_LRC << std::endl;
    resultFile << "d2UdV2_LRC:   " << std::fixed << std::setprecision(6) << d2UdV2_LRC << "\n" << std::endl;
    resultFile << result_fileheader << std::endl;
    resultFile.close();

    visFile.open(filename_vis, std::ios::out);
    visFile << "~ 1  LJ  0.0000  0.0000   0.0000  1.0000  2\n\n" << std::endl;
    visFile.close();
    writeVis();  // Write initial particle positions
}

// Thermostat (velocity scaling)
void scaleVelocity() {
    double velo_sqrt = 0.;
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            velo_sqrt += prtl_velocities[i][d] * prtl_velocities[i][d];
        }
    }
    velo_sqrt *= boxlength_sqrt;
    double temperature_avg = std::sqrt(nDims * num_prtls * temperature / velo_sqrt);
#if (DEBUG_OUT)
    std::cout << "Temperature scaling factor: " << temperature_avg << std::endl;
#endif
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            prtl_velocities[i][d] *= temperature_avg;
        }
    }
}

// Assignment of the initial velocities according to the temperature
void assignVelocity() {
    double temperature_avg = std::sqrt(nDims * temperature) / boxlength;
    double r[nDims] = {0.};
    double r_sqrd = 0.0;
    double fac = 0.;

    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            r[d] = 2. * random_num(rng) - 1.;
        }
        r_sqrd = 0.0;
        for (short d = 0; d < nDims; ++d) {
            r_sqrd += r[d] * r[d];
        }
        fac = temperature_avg / std::sqrt(r_sqrd);
        for (short d = 0; d < nDims; ++d) {
            prtl_velocities[i][d] = fac * r[d];
        }
    }

    // Set macroscopic velocity to zero
    double drift[nDims] = {0.};
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            drift[d] += prtl_velocities[i][d];
        }
    }
    for (short d = 0; d < nDims; ++d) {
        drift[d] = drift[d] / num_prtls;
    }
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            prtl_velocities[i][d] -= drift[d];
        }
    }
    // Test if macroscopic velocity is zero
    double drift_test[nDims] = {0.};
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            drift_test[d] += prtl_velocities[i][d];
        }
    }
    if (std::accumulate(drift_test, drift_test + nDims, 0.0) > 1e-6) {
        std::cout << "Drift after being set to zero ";
        for (short d = 0; d < nDims; ++d) {
            std::cout << drift_test[d] << " ";
        }
        std::cout << std::endl;
    }

    // Rescale to desired temperature
    scaleVelocity();
}

// 1st part of the Velocity-Verlet integrator: New position
void verlet1() {
    double time_var = 0.5 * delta_time * delta_time / boxlength;
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            prtl_positions[i][d] = prtl_positions[i][d] + prtl_velocities[i][d] * delta_time + prtl_forces[i][d] * time_var;
            prtl_positions[i][d] = prtl_positions[i][d] - static_cast<int>(2 * prtl_positions[i][d] - 1);  // Periodic boundary
        }
    }
}

// 2nd part of the Velocity-Verlet integrator: New velocity
void verlet2() {
    double time_var = 0.5 * delta_time / boxlength;
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            prtl_velocities[i][d] += (prtl_forces[i][d] + prtl_forces_prev[i][d]) * time_var;
        }
    }
}

// Calculation and output of state variables
void getStateValues(const int step) {
    double velo_sqrt = 0.;
    for (int i = 0; i < num_prtls; ++i) {
        for (short d = 0; d < nDims; ++d) {
            velo_sqrt += prtl_velocities[i][d] * prtl_velocities[i][d];
        }
    }

    velo_sqrt *= boxlength_sqrt;
    ekin_accum += 0.5 * velo_sqrt;
    U_accum += U_step + num_prtls * density * U_LRC;
    p_accum += virial_step / num_prtls + density * p_LRC;
    mu_accum += mu_step;
    num_test_accum += num_test_step;

    // Lustig formalism
    const double U_tot_corr = num_prtls * density * U_LRC;                   // factor N^2/V^1
    const double dUdV_tot_corr = -density * density * dUdV_LRC;              // factor N^2/V^2
    const double d2UdV2_tot_corr = density * density * d2UdV2_LRC / volume;  // factor N^2/V^3
    const double U_tot = U_step + U_tot_corr;                                // Pot. energy of whole configuration in current step
    const double dUdV_tot = dUdV + dUdV_tot_corr;
    const double d2UdV2_tot = d2UdV2 + d2UdV2_tot_corr;

#if (DEBUG_OUT)
    std::cout << " U_step " << U_step << " U_tot_corr " << U_tot_corr << std::endl;
    std::cout << " dUdV " << dUdV << " dUdV_tot_corr " << dUdV_tot_corr << std::endl;
    std::cout << " d2UdV2 " << d2UdV2 << " d2UdV2_tot_corr " << d2UdV2_tot_corr << std::endl;
#endif

    // Accumulate values
    dUdV_accum += dUdV_tot;
    d2UdV2_accum += d2UdV2_tot;
    U2_accum += U_tot * U_tot;
    U3_accum += U_tot * U_tot * U_tot;
    dUdV_2_accum += dUdV_tot * dUdV_tot;
    U_dUdV_accum += U_tot * dUdV_tot;
    U_2_dUdV_accum += U_tot * U_tot * dUdV_tot;
    U_dUdV_2_accum += U_tot * dUdV_tot * dUdV_tot;
    U_d2UdV_2_accum += U_tot * d2UdV2_tot;

    // Write out data
    if ((step % writefreq_output) == 0) {
        const long long step_LL = static_cast<long long>(step);  // Convert to long long to prevent overflow
        const double step_numPrtls_inv = 1. / (step_LL * num_prtls);
        const double temperature_avg = 2. * ekin_accum * step_numPrtls_inv / nDims;
        const double U_avg = U_accum * step_numPrtls_inv;
        const double ekin_avg = ekin_accum * step_numPrtls_inv;
        const double etotal_avg = U_avg + ekin_avg;
        const double pressure_avg = temperature_avg * density + density * p_accum / step;
        double chemPot_res_avg = 0.0;
        double numTest = 0.0;
        if ((mu_accum > 0.0) && (num_test_accum > 0ul)) {
            numTest = static_cast<double>(num_test_accum) / step;
            chemPot_res_avg = -std::log(mu_accum / num_test_accum) + std::log(density);  // Implemented in accordance to ms2
#if (DEBUG_OUT)
            std::cout << "mu_accum " << mu_accum << " num_test_accum " << num_test_accum << " mu_accum/num_test_accum " << mu_accum / num_test_accum << " std::log(mu_accum/num_test_accum) "
                      << std::log(mu_accum / num_test_accum) << " std::log(density) " << std::log(density) << " chemPot_res_avg " << chemPot_res_avg << " numTest " << numTest << std::endl;
#endif
        }

        // Lustig formalism
        const double U_temp = U_accum / step;
        const double dUdV_temp = dUdV_accum / step;
        const double d2UdV2_temp = d2UdV2_accum / step;
        const double U2_temp = U2_accum / step;
        const double U3_temp = U3_accum / step;
        const double dUdV_2_temp = dUdV_2_accum / step;
        const double U_dUdV_temp = U_dUdV_accum / step;
        const double U_2_dUdV_temp = U_2_dUdV_accum / step;
        const double U_dUdV_2_temp = U_dUdV_2_accum / step;
        const double U_d2UdV_2_temp = U_d2UdV_2_accum / step;

        const double num_prtls_inv = 1. / static_cast<double>(num_prtls);
        const double beta = 1. / temperature_avg;
        const double beta2 = beta * beta;
        const double beta3 = beta * beta * beta;
        const double rho_inv = 1. / density;
        const double rho_inv_sqrt = rho_inv * rho_inv;

        const double A10r = beta * U_temp * num_prtls_inv;
        const double A01r = -1. * beta * rho_inv * dUdV_temp;

        const double A20r = beta2 * num_prtls_inv * (U_temp * U_temp - U2_temp);
        const double A11r = -1. * rho_inv * beta * dUdV_temp + rho_inv * beta2 * U_dUdV_temp - rho_inv * beta2 * U_temp * dUdV_temp;
        const double A02r =
            rho_inv_sqrt * num_prtls * beta * d2UdV2_temp - rho_inv_sqrt * num_prtls * beta2 * dUdV_2_temp + rho_inv_sqrt * num_prtls * beta2 * dUdV_temp * dUdV_temp + 2. * rho_inv * beta * dUdV_temp;

        const double A30r = beta3 * num_prtls_inv * (U3_temp - 3. * U_temp * U2_temp + 2. * U_temp * U_temp * U_temp);
        const double A21r = 2. * rho_inv * beta2 * U_dUdV_temp - 2. * rho_inv * beta2 * U_temp * dUdV_temp + rho_inv * beta3 * U2_temp * dUdV_temp - rho_inv * beta3 * U_2_dUdV_temp +
                            2. * rho_inv * beta3 * U_temp * U_dUdV_temp - 2. * rho_inv * beta3 * U_temp * U_temp * dUdV_temp;

        const double A12r = rho_inv_sqrt * num_prtls * beta3 * U_dUdV_2_temp + 2. * rho_inv_sqrt * num_prtls * beta3 * U_temp * dUdV_temp * dUdV_temp -
                            rho_inv_sqrt * num_prtls * beta3 * U_temp * dUdV_2_temp - 2. * rho_inv_sqrt * num_prtls * beta3 * U_dUdV_temp * dUdV_temp +
                            2. * rho_inv_sqrt * num_prtls * beta2 * dUdV_temp * dUdV_temp + rho_inv_sqrt * num_prtls * beta2 * U_temp * d2UdV2_temp -
                            2. * rho_inv_sqrt * num_prtls * beta2 * dUdV_2_temp - rho_inv_sqrt * num_prtls * beta2 * U_d2UdV_2_temp + rho_inv_sqrt * num_prtls * beta * d2UdV2_temp +
                            2. * rho_inv * beta2 * U_temp * dUdV_temp - 2. * rho_inv * beta2 * U_dUdV_temp + 2. * rho_inv * beta * dUdV_temp;

        std::ofstream resultFile;
        resultFile.open(filename_result, std::ios::app);

        auto format_helper = [&](auto val) {
            std::ostringstream oss;
            oss << std::fixed << std::setw(file_columnwidth) << std::setprecision(6) << val << " ";
            return oss.str();
        };
        
        resultFile
            << format_helper(step)
            << format_helper(temperature_avg)
            << format_helper(density)
            << format_helper(pressure_avg)
            << format_helper(dUdV_temp)
            << format_helper(U_avg)
            << format_helper(ekin_avg)
            << format_helper(etotal_avg)
            << format_helper(chemPot_res_avg)
            << format_helper(numTest)
            << format_helper(chemPot_res_avg - A01r - std::log(density))  // A00r
            << format_helper(A10r)
            << format_helper(A01r)
            << format_helper(A20r)
            << format_helper(A11r)
            << format_helper(A02r)
            << format_helper(A30r)
            << format_helper(A21r)
            << format_helper(A12r)
            << std::endl;
        resultFile.close();

        if ((step == steps_prod) && (!flg_equi)) {
            std::ofstream finResFile;
            finResFile.open(filename_final, std::ios::out);
            finResFile << "dimensions   " << nDims << std::endl;
#if (LJTS)
            finResFile << "fluid        LJTS" << std::endl;
#else
            finResFile << "fluid        LJfull" << std::endl;
#endif
            finResFile << "numParticles " << num_prtls << std::endl;
            finResFile << "boxlength    " << std::setprecision(8) << boxlength << std::endl;
            finResFile << "cutoff       " << std::setprecision(6) << r_cutoff << std::endl;
            finResFile << "timestepEqui " << steps_equi << std::endl;
            finResFile << "timestepProd " << steps_prod << std::endl;
            finResFile << "temperature  " << std::setprecision(8) << temperature_avg << std::endl;
            finResFile << "density      " << std::setprecision(8) << density << std::endl;
            finResFile << "pressure     " << std::setprecision(8) << pressure_avg << std::endl;
            finResFile << "energyPot    " << std::setprecision(8) << U_avg << std::endl;
            finResFile << "energyKin    " << std::setprecision(8) << ekin_avg << std::endl;
            finResFile << "energyTot    " << std::setprecision(8) << etotal_avg << std::endl;
            finResFile << "mu_res       " << std::setprecision(8) << chemPot_res_avg << std::endl;
            finResFile << "numTestMu    " << std::setprecision(8) << numTest << std::endl;
            finResFile << "A10r    " << std::setprecision(8) << A10r << std::endl;
            finResFile << "A01r    " << std::setprecision(8) << A01r << std::endl;
            finResFile << "A20r    " << std::setprecision(8) << A20r << std::endl;
            finResFile << "A11r    " << std::setprecision(8) << A11r << std::endl;
            finResFile << "A02r    " << std::setprecision(8) << A02r << std::endl;
            finResFile << "A30r    " << std::setprecision(8) << A30r << std::endl;
            finResFile << "A21r    " << std::setprecision(8) << A21r << std::endl;
            finResFile << "A12r    " << std::setprecision(8) << A12r << std::endl;
            finResFile.close();
        }
    }
}

// Reset variables for calculating the state variables after the equilibration phase
void resetValues() {
    ekin_accum = 0.0;
    U_accum = 0.0;
    p_accum = 0.0;
    mu_accum = 0.0;
    num_test_accum = 0;

    std::fill(count_RDF, count_RDF + num_shells_RDF, 0);

    // Lustig formalism
    dUdV_accum = 0.;
    d2UdV2_accum = 0.;
    U2_accum = 0.;
    U3_accum = 0.;
    dUdV_2_accum = 0.;
    U_dUdV_accum = 0.;
    U_2_dUdV_accum = 0.;
    U_dUdV_2_accum = 0.;
    U_d2UdV_2_accum = 0.;

    std::ofstream resultFile;
    resultFile.open(filename_result, std::ios::app);
    resultFile << "\n" << result_fileheader << std::endl;
    resultFile.close();
}

// Calculation and output of RDF
void writeRDFs(const int step) {
    std::ofstream rdfFile;
    rdfFile.open(filename_RDF, std::ios::out);
    rdfFile << "       Index      Radius        g(r)" << std::endl;
    double gamma_pi_factor = (2. / nDims) * (std::pow(std::numbers::pi, 0.5 * nDims) / gamma_factor());
    for (uint i = 0; i < num_shells_RDF; ++i) {
        const double radius_i = (i + 0.5) * r_cutoff / num_shells_RDF;
        const double dVolume = gamma_pi_factor * std::pow(r_cutoff / num_shells_RDF, nDims) * (std::pow(i + 1, nDims) - std::pow(i, nDims));  // Volume of the n-dimensional spherical shell
        double G = static_cast<double>(count_RDF[i]) / (dVolume * density);                                                                   // Particles were counted twice due to parallelization
        G /= (static_cast<double>(num_prtls) * step);  // Average over production run steps and particles; use cast to prevent overflow
        rdfFile << std::fixed << std::setw(12) << std::setprecision(6) << i << std::fixed << std::setw(12) << std::setprecision(6) << radius_i << std::fixed << std::setw(12) << std::setprecision(6)
                << G << std::endl;
    }
    rdfFile.close();
}

// Final steps of the simulation
void finalizeSimulation() {
    std::ofstream visFile;
    visFile.open(filename_vis, std::ios::app);
    visFile << "\n\n##" << std::endl;
    visFile.close();
}

// ----- MAIN PROGRAM -----

int main() {
    time_t time_start = std::time(0);
    // Random seed for random number generation; this allows simulations to be replicated exactly
    //rng.seed(static_cast<unsigned int>(time_start));
    rng.seed(123);

    std::cout << "Start of simulation (MD) with " << omp_get_max_threads() << " threads" << std::endl;

    // Calculation of the long-range corrections
    calcCutoffCorrections();

    // Resize vectors
    prtl_positions.resize(num_prtls);
    prtl_velocities.resize(num_prtls);
    prtl_forces.resize(num_prtls);
    prtl_forces_prev.resize(num_prtls);

    // Setup of initial configuration
    std::cout << "Starting with lattice" << std::endl;
    std::cout << "Size of box in one direction: " << boxlength << std::endl;
    std::cout << "Volume of box: " << volume << std::endl;
    std::cout << "Number of molecules: " << num_prtls << std::endl;
    std::cout << "Cutoff radius: " << r_cutoff << std::endl;
    std::cout << "Temperature: " << temperature << std::endl;
    std::cout << "Density: " << density << std::endl;

    // Setup lattice
    makeLattice();
    // Assignment of initial velocities and scaling to temperature
    assignVelocity();

    // Resize cell vectors
    cell_index.resize(num_prtls);
    std::fill(cell_index.begin(), cell_index.end(), -1);

    std::cout << "Number of cells in one direction: " << grid_n_part << std::endl;
    std::cout << "Number of neighbor cells in one direction: " << numCellsDirect1D << std::endl;
    std::cout << "Normalized width of one cell: " << grid_ddims << std::endl;
    std::cout << "ncells: " << ncells << std::endl;
    std::cout << "avg mols/cell " << static_cast<float>(num_prtls) / ncells << std::endl;

    if (grid_n_part <= 1) {
        std::cerr << "ERROR: At least 2 cells in each direction are required!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (fabs(1.0 - grid_n_part * grid_ddims) > 1e-6) {
        std::cerr << "Problem detected: grid_n_part*grid_ddims = " << grid_n_part*grid_ddims << " != 1.0"  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (r_cutoff > 0.5 * boxlength) {
        std::cerr << "ERROR: Cutoff (" << r_cutoff << ") must be smaller than half boxlength (" << 0.5 * boxlength << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Create output files
    initFiles();

    // init cell lists
    std::vector<int> vtmp;
    for (int i = 0; i < ncells; ++i) {
        cell_list.push_back(vtmp);
        cell_list[i].push_back(i);
    }

    cell_list_clear();
    cell_list_compute_c();
    cell_list_build();

    writeVis();

    std::cout << "Calculating initial forces..." << std::endl;
    calcPotForceLJ();

    if (flg_equi) {
        // Equilibration run
        for (int step = 1; step <= steps_equi; ++step) {
            if ((step % 1000) == 0) {
                std::cout << "Equi: Timestep = " << step << std::endl;
            }
            verlet1();
            cell_list_clear();
            cell_list_compute_c();
            cell_list_build();
            calcPotForceLJ();
            verlet2();
            // No sampling of chemical potential during equilibration phase
            getStateValues(step);
            if (flg_ensemble_NVT) {
                scaleVelocity();
            }
            if ((step % writefreq_vis_RDF) == 0) {
                writeVis();
            }
        }
        // Reset variables for calculating the state variables after the equilibration phase
        resetValues();
    }

    // Production run
    for (int step = 1; step <= steps_prod; ++step) {
        if ((step % 1000) == 0) {
            std::cout << "Prod: Timestep = " << step << std::endl;
        }
        verlet1();
        cell_list_clear();
        cell_list_compute_c();
        cell_list_build();
        calcPotForceLJ();
        verlet2();
        if (flg_chemPot) {
            calcChemicalPotential();
        }
        getStateValues(step);
        if (flg_ensemble_NVT) {
            scaleVelocity();
        }
        if ((step % writefreq_vis_RDF) == 0) {
            writeVis();
            writeRDFs(step);
        }
    }

    writeRDFs(steps_prod);
    finalizeSimulation();

    time_t time_end = std::time(0);
    double runtime_hours = std::difftime(time_end, time_start) / 3600.;

    std::cout << "End of simulation (MD) after " << std::setprecision(4) << runtime_hours << " hours ( " << runtime_hours * omp_get_max_threads() << " core-hours )" << std::endl;

    return EXIT_SUCCESS;  // Success
}
