################################################################################
# Load packages
################################################################################

using FFTW
using LinearAlgebra
using Statistics
using Dates
using JLD2
using PyPlot
using Printf
using Random: seed!
using SparseArrays
using Base.Threads
using LoopVectorization

################################################################################
# Geometry
################################################################################

Nx = 128
Ny = 128*2
Lx = 50/2 #   500e3 * Nx
Ly = 50 #   500e3 * Ny
dx = Lx / Nx
dy = Ly / Ny

# Do not change these vvv
x = collect(0:dx:Lx-dx)
y = collect(0:dy:Ly-dy)

################################################################################
# Model params
################################################################################

f0 = 1.0 # 1e-4
beta = 0.25 # 1.4e-11; beta promotes unstable edge waves
g = 1 # 9.81
H1 = 20.0 # 2000.0
H2 = 20.0 # 2000.0
ρ0 = 1.0
Δρ = 0.1  # 0.75

U0 = 1.0  # Background shear (m/s)

# Do not change these vvv
gprime = g * Δρ / ρ0

F1 = f0^2 / (gprime * H1)
F2 = f0^2 / (gprime * H2)

Ld = sqrt((H1+H2) * gprime) / 2 / f0    # for beta=0.25 and U0=1, LSA says Ld \leq 2 will produce BCI (if U1 were constant, of course)


################################################################################
# Timestepping params
################################################################################

cfl = 0.05      # nominal CFL

dt = cfl * minimum([dx, dy]) / U0       # time step
nt = 15000                              # number of time steps

timestep_method = "RK4" # "RK4_int"     # options are: RK4, RK4_int

################################################################################
# Load source code
################################################################################

src_dir = "/home/matt/Desktop/research/QG/QG_channel/src/"
src_files = readdir(src_dir)
for file in src_files include(src_dir*file) end

################################################################################
# Define background flow profile
################################################################################

WC = 4  # Width of boundary where background flow decays to zero (max of 0.5)

ψ1_bg, U_bg = Lee1997_bg_jet(U0, WC)

ψ1_bg = ψ1_bg .* ones(Nx, Ny)
ψ2_bg = zeros(Nx, Ny)

ψ_diff_bg = ψ1_bg .- ψ2_bg


################################################################################
# Define paths for saving streamfunction files and figures; and frequency of output
################################################################################

# this saves meridional bands (full zonal extent) of i) ψ1, ii) ψ2, and iii) t
save_bool = true
save_path = "/home/matt/Desktop/research/QG/QG_channel_output/data/L97_WC4_BCI/"
y_width = 0.75  # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain
save_every = round(Int,nt/20)      # period of save frequency


# this saves full streamfunction field (and time) at end of simulation run, in case you'd like to use the field as initial conditions later on
# directory is defined by save_path (above)
save_last = true


# this plots panels at fig_path; the plot function (defined in output_fcns.jl) can be modified to be whatever you want to see
plot_basic_bool = false
plot_BCI_bool = true
fig_path = "/home/matt/Desktop/research/QG/QG_channel_output/anim/L97_WC4_BCI/"
plot_every = round(Int,nt/400)      # period of plot output frequency


################################################################################
# Damping (biharmonic viscosity, linear friction, and thermal damping)
################################################################################
ν = 1e-3 #  0.01 * dx^4 / dt # 1e6          # Hyperviscosity (m⁴/s)  L97 uses 6e-3

r = 0.05         # Ekman friction (1/s)  L97 uses 0.1
α = 30^-1        # Thermal damping (1/s)

################################################################################
# Restart from L97_WC4_init.jl run
################################################################################

restart_path = "/home/matt/Desktop/research/QG/QG_channel_output/data/L97_WC4_init/"


## don't change below here vvv
ψfiles = readdir(restart_path)

t_array = define_t_of_saved_files(Ny, restart_path)

t0 = t_array[end]

file_start, file_end = extract_time(ψfiles[1]; file_strings=true)

savedata = load(restart_path * file_start * string(t_array[end]) * file_end)

ψ1_restart = savedata["jld_data"]["ψ1"]
ψ2_restart = savedata["jld_data"]["ψ2"]

q1, q2 = compute_qg_pv(ψ1_restart, ψ2_restart)

################################################################################
# Run model from initial conditions
################################################################################
include("../define_vars.jl")
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1, H2], ρ0, Δρ, ν, r, α, U0, WC)

run_model(q1, q2, t0, params)


# now you can run L97_WC20_SS.jl to calculate steady-state turbulent statistics