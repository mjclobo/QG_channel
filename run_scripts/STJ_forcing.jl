################################################################################
# Load packages & Source
################################################################################
using FFTW, LinearAlgebra, Statistics, Dates, JLD2, PyPlot, Printf
using Random: seed!
using SparseArrays, Base.Threads, LoopVectorization, Parameters, KernelAbstractions

################################################################################
# 1. Core Parameters & Geometry
################################################################################
# Geometry and Grid (Atmospheric Earth-scale Channel)
Nx, Ny, Nz = 64, 128, 2
Lx, Ly = 25000e3, 50000e3   # 

dx = Lx / (Nx)
dy = Ly / (Ny)
x = collect(range(0, Lx, length=Nx))
y = collect(range(-Ly/2, Ly/2, length=Ny))

# Physics (Atmospheric Parameters)
beta = 1.6e-11            # Mid-latitude beta
f0, g, ρ0 = 1.0e-4, 9.81, 1.2  # Air density = 1.2 kg/m^3
H1, H2 = 5000.0, 5000.0   # 10km total troposphere height, split evenly

# Back-calculate g' to exactly match Ld = 1000 km (Atmospheric scale)
Ld = 800e3
gprime = (Ld * f0)^2 * (H1 + H2) / (H1 * H2)
Δρ = gprime * ρ0 / g

# Jet Parameters (Internal Shear Forcing)
U0 = 30 #25                  # Set to 0.0 for STJ forced-only runs
trans = 0.0
WC = 10000e3                
y0_EDJ = 0 # 2500e3

F1 = 2 * f0^2 / (gprime * H1)
F2 = 2 * f0^2 / (gprime * H2)

# Timestepping
cfl = 0.4
U_scale = 30.0            # Used for CFL since U0 is 0.0
dt = cfl * minimum([dx, dy]) / U_scale
ndays = 200
nt = round(Int, (ndays * 24 * 3600) / dt)

# Damping
N_steps = 10.0
ν = 10 * (dx^4) / (N_steps * dt * (2π)^4)
r = 1.0 / (10.0 * 24 * 3600)       # 5-day Ekman bottom friction 
α = 1.0 / (30.0 * 24 * 3600)      # Thermal relaxation

# Toggle to fix the background flow (True = fixed U as U_bg, False = evolving jet)
fix_zonal_mean = false

if fix_zonal_mean == true
    # Global α override for the fixed background framework to prevent perturbation damping
    α = 0.0
end


###########################################################
## Load source code
###########################################################
src_dir = "/home/matt/Desktop/research/QG/2LQG/HeldLobo_proj/QG_channel/src/"
include(src_dir * "../define_vars.jl") # Ensure ModelParams struct is available
for file in readdir(src_dir)
    if startswith(file, ".")
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
# ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = Lee1997_bg_jet(U0, WC; σ=50.0e3)
ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = blended_transport_jet(y; y0=y0_EDJ, σ=50.0e3, W=WC, trans=trans, U0=U0)

# Expand the 1D profile to an Nx by Ny matrix
ψ1_bg = repeat(ψ1_bg_1D', Nx, 1) 
ψ2_bg = zeros(Nx, Ny)
ψ_diff_bg = ψ1_bg .- ψ2_bg

bg_of_t = false
function ψ_diff_bg_of_t(t; t0=250)
    pd, A0, A1 = 250, 11, 3
    t < 250 ? Lee1997_bg_jet(1.0, A0) : Lee1997_bg_jet(1.0, A0 + A1 * sin(2 * pi * (t - t0) / pd))
end


#########################################################################
## STJ forcing (Atmospheric Scales) - CORRECTED BOUNDARIES
###################################################################
# 1. Define the target velocity
subtrop_y = -9000e3     
width = 5000e3          
U_STJ_target = @. 25.0 * exp(-((y - subtrop_y)^2) / (2 * width^2))

# Flat-top taper: 1.0 everywhere, smoothly curving to 0 only at the last 100km
taper_window = ones(Ny)
sponge_L = 100e3 # only taper the 100km closest to the walls
for i in 1:Ny
    dist_south = y[i] - y[1]
    dist_north = y[end] - y[i]
    if dist_south < sponge_L
        taper_window[i] = sin((pi/2) * dist_south / sponge_L)^2
    elseif dist_north < sponge_L
        taper_window[i] = sin((pi/2) * dist_north / sponge_L)^2
    end
end
U_STJ_target .*= taper_window
U_STJ_target .-= mean(U_STJ_target)
# 2. Numerically integrate U to get the target streamfunction (ψ = -∫ U dy)
ψ_STJ_target = zeros(Ny)
for i in 2:Ny
    ψ_STJ_target[i] = ψ_STJ_target[i-1] - U_STJ_target[i] * dy
end

# Center the streamfunction to prevent domain-wide mean offsets
ψ_STJ_target .-= mean(ψ_STJ_target)

# 3. Use your exact model operators to compute the target PV
ψ2_STJ_target = zeros(Ny) 
q1_STJ_target, q2_STJ_target = compute_qg_pv_bar(ψ_STJ_target, ψ2_STJ_target; lap_op=ops.L1D)

# 4. NEW: Create a spatially dependent nudging timescale
α_STJ_max = 1.0 / (30.0 * 24.0 * 3600.0)

# Create a Gaussian mask centered on the STJ that decays to zero in the mid-latitudes
# Making the mask slightly wider than the jet (e.g., 1.5 * width) ensures a smooth transition
α_STJ_1D = @. α_STJ_max * exp(-((y - subtrop_y)^2) / (2 * width^2))

# (If your code requires a 2D array in the RHS function, just broadcast it):
# α_STJ_2D = repeat(α_STJ_1D', Nx, 1)


################################################################################
# 3. Output & Diagnostic Configuration
################################################################################
base_name = "atmos_WC_init_beta$(beta)_WC$(WC)_y0EDJ$(y0_EDJ)_r$(r)_h0topo$(h0_topo)_Wshelf$(W_shelf)_U0$(U0)_tau0$(τ0)_alpha$(α)"
base_dir = "/home/matt/Desktop/research/QG/2LQG/HeldLobo_proj/QG_channel_output"

# Define frequencies as standalone variables first
save_every = round(Int, nt/20)
plot_every = round(Int, nt/60)
diag_every = round(Int, nt/500)

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

# 1. Instantiate the operators first so o.L1D is available for PV calculations
ops = QGOperators(solver2L, A_lu, rhs_pa, ψ_vec, L1D, L2D, zeros(Nx, Ny), zeros(Nx, Ny))

# 2. Compute 1D background profiles using the operator from ops
ψ2_bg_1D = zeros(Ny)
q1_bg_1D, q2_bg_1D = compute_qg_pv_bar(ψ1_bg_1D, ψ2_bg_1D; lap_op=ops.L1D)

# 3. Create initial 2D random perturbations
q1_prime = 1.0e-5 * randn(Nx, Ny)   
q2_prime = 1.0e-5 * randn(Nx, Ny)   

# Enforce no-flow channel wall boundaries on the noise matrices
q1_prime[:, 1] .= 0.0; q1_prime[:, end] .= 0.0
q2_prime[:, 1] .= 0.0; q2_prime[:, end] .= 0.0

# 4. Build the unified QGState and pack into QGModel
state = QGState(
    q1_prime, q2_prime, zeros(Ny), zeros(Ny), 
    zeros(Nx, Ny), zeros(Nx, Ny), zeros(Ny), zeros(Ny)
)
model = QGModel(state, ops)

################################################################################
# 5. Diagnostic Suite Setup
################################################################################
suite = DiagnosticSuite()
# add!(suite, :Pseudomomentum, PseudomomentumDiag(nt, dt, diag_every, Ny))
add!(suite, :ZonalMeanEnergy, ZonalMeanEnergyDiag(nt, dt, diag_every, Ny))
# add!(suite, :ScalarEnergy, ScalarEnergyDiag(nt, dt, diag_every, zone_start_ind, zone_end_ind))
add!(suite, :HovmollerZonalFlow, HovmollerZonalFlowDiag(nt, dt, diag_every, Ny))

################################################################################
# 6. Run Model Configuration
################################################################################
t0 = 0.0

# Run the unified simulation framework
run_model_decomp(
    model, suite, out_cfg, ψ_diff_bg, U_bg, t0, params; 
    q1_STJ_target=q1_STJ_target, q2_STJ_target=q2_STJ_target, t_start_diag=0, 
    output_every=250, fix_zonal_mean=fix_zonal_mean
)



