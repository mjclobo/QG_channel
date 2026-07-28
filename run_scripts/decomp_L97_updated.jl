################################################################################
# Load packages & Source
################################################################################
using FFTW, LinearAlgebra, Statistics, Dates, JLD2, PyPlot, Printf
using Random: seed!
using SparseArrays, Base.Threads, LoopVectorization, Parameters, KernelAbstractions

################################################################################
# 1. Core Parameters & Geometry (Non-Dimensional)
################################################################################
Nx, Ny, Nz = 64, 128, 2
Lx, Ly = 64.0, 128.0

dx = Lx / (Nx)
dy = Ly / (Ny)
x = collect(range(0, Lx, length=Nx))
y = collect(range(-Ly/2, Ly/2, length=Ny))

# Physics
beta = 0.25
f0, g, ρ0 = 1.0, 1.0, 1.0
H1, H2 = 20.0, 20.0
Δρ = 0.1

U0 = 1.0
trans = 1.0
WC = 35.0

gprime = g * Δρ / ρ0
F1 = 2 * f0^2 / (gprime * H1)
F2 = 2 * f0^2 / (gprime * H2)

Ld = sqrt((H1+H2) * gprime) / (2 * f0) 

# Timestepping
cfl = 0.04
dt = cfl * minimum([dx, dy]) / U0
ndays = 200
nt = round(Int, ndays / dt)
timestep_method = "RK4"

# Damping
N_steps = 10.0
ν = 10 * (dx^4) / (N_steps * dt * (2π)^4)
r = 0.1
α = 1.0 / 30.0

fix_zonal_mean = false
if fix_zonal_mean == true
    α = 0.0
end

###########################################################
## Load source code
###########################################################
src_dir = "/home/matt/Desktop/research/QG/2LQG/HeldLobo_proj/QG_channel/src/"
include(src_dir * "../define_vars.jl") # Ensure ModelParams struct is available
for file in readdir(src_dir)
    # Exclude the 2LQG unified main file as per the old decomposed script logic
    if startswith(file, ".") || startswith(file, "2LQG")
        #do nothing
    else
       include(joinpath(src_dir, file))
    end
end

# Package everything cleanly into params struct immediately
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1, H2], ρ0, Δρ, ν, r, α, U0, WC, trans)

################################################################################
# 2. Background Flow & Topography
################################################################################
ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = Lee1997_bg_jet(U0, WC; σ=5.0)

# Expand the 1D profile to an Nx by Ny matrix
ψ1_bg = repeat(ψ1_bg_1D', Nx, 1) 
ψ2_bg = zeros(Nx, Ny)
ψ_diff_bg = ψ1_bg .- ψ2_bg

bg_of_t = false
function ψ_diff_bg_of_t(t; t0=250)
    pd, A0, A1 = 250, 11, 3
    t < 250 ? Lee1997_bg_jet(1.0, A0) : Lee1997_bg_jet(1.0, A0 + A1 * sin(2 * pi * (t - t0) / pd))
end

U_STJ_target=0.0

################################################################################
# 3. Output & Diagnostic Configuration
################################################################################
base_name = "nondim_beta$(beta)_WC$(WC)_trans$(trans)_r$(r)"
base_dir = "/home/matt/Desktop/research/QG/2LQG/HeldLobo_proj/QG_channel_output"

# Define frequencies
save_every = round(Int, nt/20)
plot_every = round(Int, nt/60)
diag_every = round(Int, nt/100)

out_cfg = OutputConfig(
    save_bool = true,
    save_last = true,
    plot_basic_bool = true,
    save_every = save_every,
    plot_every = plot_every,
    diag_every = diag_every,
    save_path = joinpath(base_dir, "data", base_name),
    fig_path  = joinpath(base_dir, "anim", base_name),
    diag_dir  = joinpath(base_dir, "diagnostics", base_name)
)

# Ensure directories exist
mkpath(out_cfg.save_path)
mkpath(out_cfg.fig_path)
mkpath(out_cfg.diag_dir)

################################################################################
# 4. Operators & State Initialization
################################################################################
seed!(1234)

# 1. Instantiate the operators
ops = QGOperators(solver2L, A_lu, rhs_pa, ψ_vec, L1D, L2D, zeros(Nx, Ny), zeros(Nx, Ny))

# 2. Create initial 2D random perturbations (matching original 1e2 * 1e-2 scaling)
q1_prime = 1.0 * randn(Nx, Ny)   
q2_prime = 1.0 * randn(Nx, Ny)   

# Enforce no-flow channel wall boundaries on the noise matrices
q1_prime[:, 1] .= 0.0; q1_prime[:, end] .= 0.0
q2_prime[:, 1] .= 0.0; q2_prime[:, end] .= 0.0

# 3. Build the unified QGState and pack into QGModel
state = QGState(
    q1_prime, q2_prime, zeros(Ny), zeros(Ny), 
    zeros(Nx, Ny), zeros(Nx, Ny), zeros(Ny), zeros(Ny)
)
model = QGModel(state, ops)

################################################################################
# 5. Diagnostic Suite Setup
################################################################################
suite = DiagnosticSuite()
add!(suite, :ZonalMeanEnergy, ZonalMeanEnergyDiag(nt, dt, diag_every, Ny))
add!(suite, :HovmollerZonalFlow, HovmollerZonalFlowDiag(nt, dt, diag_every, Ny))

################################################################################
# 6. Run Model Configuration
################################################################################
t0 = 0.0

# Run the unified simulation framework
run_model_decomp(
    model, suite, out_cfg, ψ_diff_bg, U_bg, t0, params; 
    t_start_diag=250, output_every=250, fix_zonal_mean=fix_zonal_mean,
    save_ind_start=zone_start_ind, save_ind_end=zone_end_ind
)



