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

Nx = 64
Ny = 64*2
Lx = 50/2 
Ly = 50 
dx = Lx / Nx
dy = Ly / Ny

# Do not change these vvv
x = collect(0:dx:Lx-dx)
y = collect(0:dy:Ly-dy)

################################################################################
# Model params
################################################################################

beta = 0.5 #

# Do not change these vvv
f0 = 1.0 # 1e-4
g = 1 # 9.81
H1 = 20.0 # 2000.0
H2 = 20.0 # 2000.0
ρ0 = 1.0
Δρ = 0.1  # 0.75

U0 = 1.0  # Background shear (m/s)

gprime = g * Δρ / ρ0

F1 = 2 * f0^2 / (gprime * H1)
F2 = 2 * f0^2 / (gprime * H2)

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

# ψ1_bg = ψ1_bg
ψ2_bg = zeros(size(ψ1_bg))

ψ_diff_bg = ψ1_bg .- ψ2_bg

################################################################################
# Define paths for saving streamfunction files and figures; and frequency of output
################################################################################

# this saves meridional bands (full zonal extent) of i) ψ1, ii) ψ2, and iii) t
save_bool = true
save_path = "/home/matt/Desktop/research/QG/QG_channel_output/data/L97_WC4_init/"
y_width = 0.75  # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain
save_every = round(Int,nt/20)      # period of save frequency


# this saves full streamfunction field (and time) at end of simulation run, in case you'd like to use the field as initial conditions later on
# directory is defined by save_path (above)
save_last = true


# this plots panels at fig_path; the plot function (defined in output_fcns.jl) can be modified to be whatever you want to see
plot_basic_bool = true
plot_BCI_bool = false
fig_path = "/home/matt/Desktop/research/QG/QG_channel_output/anim/L97_WC4_BCI/"
plot_every = round(Int,nt/400)      # period of plot output frequency


################################################################################
# Damping (biharmonic viscosity, linear friction, and thermal damping)
################################################################################
ν = 1e-3 #  0.01 * dx^4 / dt # 1e6          # Hyperviscosity (m⁴/s)  L97 uses 6e-3

r = 0.05         # Ekman friction (1/s)  L97 uses 0.1
α = 30^-1        # Thermal damping (1/s)

################################################################################
# Set initial conditions
################################################################################

ψ1_bar = zeros(size(ψ1_bg)) # zeros(Nx, Ny) # 
ψ2_bar = zeros(size(ψ2_bg))

q1_bar = zeros(size(ψ1_bg))
q2_bar = zeros(size(ψ2_bg))

q1_bg, q2_bg = compute_qg_pv_bar(ψ1_bg, ψ2_bg)
# q1_bar, q2_bar = compute_qg_pv_bar(ψ1_bg, ψ2_bg)

# seed!(2222)
seed!(1234)

# q1 .+= 1e-2 * randn(Nx, Ny)
# q2 .+= 1e-2 * randn(Nx, Ny)

q1_prime = 1e-2 * randn(Nx, Ny)    #  zeros((Nx, Ny))  #
q2_prime = 1e-2 * randn(Nx, Ny)    # zeros((Nx, Ny))  # 

t0 = 0    # initial timestamp (in seconds)

################################################################################
# Run model from initial conditions
################################################################################
include("../define_vars.jl")
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1, H2], ρ0, Δρ, ν, r, α, U0, WC)

run_model_decomp(q1_bar, q2_bar, q1_prime, q2_prime, t0, params)


# now you can run L97_WC4_SS.jl to calculate steady-state turbulent statistics