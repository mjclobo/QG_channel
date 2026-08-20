################################################################################
# Load packages & Source
################################################################################
using FFTW, LinearAlgebra, Statistics, Dates, JLD2, PyPlot, Printf
using Random: seed!
using SparseArrays, Base.Threads, LoopVectorization, Parameters, KernelAbstractions

################################################################################
# 1. Core Parameters & Geometry
################################################################################
# Geometry and Grid (Matches the 3.125 km resolution of the paper)
Nx, Ny, Nz = 120, 190, 2
Lx, Ly = 1200e3/2, 1900e3/2

dx = Lx / (Nx)
dy = Ly / (Ny)
x = collect(range(0, Lx, length=Nx))
y = collect(range(-Ly/2, Ly/2, length=Ny))

# Physics (Oceanic Parameters)
beta = 0. # 1.5e-11
f0, g, ρ0 = -1.0e-4, 9.8, 1025.0  # Southern Hemisphere Coriolis
H1, H2 = 1000.0, 3000.0            # 2000m total depth

# Back-calculate g' to exactly match Ld = 11.5 km from the paper
Ld = 11.5e3
gprime = (Ld * f0)^2 * (H1 + H2) / (H1 * H2)
Δρ = gprime * ρ0 / g

# Jet Parameters (Internal Shear Forcing)
U0 = 0.2                          # Internal shear forcing (0.5 m/s oceanic jet)
trans = 1.0
WC = 100e3                        # half-width baroclinic zone

F1 = 2 * f0^2 / (gprime * H1)
F2 = 2 * f0^2 / (gprime * H2)

# Timestepping
cfl = 0.1
dt = cfl * minimum([dx, dy]) / U0
ndays = 250
nt = round(Int, (ndays * 24 * 3600) / dt)

# Damping
N_steps = 10.0
ν = 10 * (dx^4) / (N_steps * dt * (2π)^4)
r = 1.0 / (10.0 * 24 * 3600)      # Oceanic bottom friction 
α = 1.0 / (10.0 * 24 * 3600)      # Thermal relaxation

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

## Jet Parameters for the Bickley Jet
# WC is jet width
y_U = 0.0      # Central latitude of the maximum baroclinic shear

# Generate the analytical background profiles
#ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = bickley_jet(y; U0=U0, L_U=WC, y_U=y_U)
ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = blended_transport_jet(y; σ=50.0e3, W=WC, trans=1.0, U0=U0)

# Expand the 1D profile to an Nx by Ny matrix
ψ1_bg = repeat(ψ1_bg_1D', Nx, 1) 
ψ2_bg = zeros(Nx, Ny)
ψ_diff_bg = ψ1_bg .- ψ2_bg

bg_of_t = false
function ψ_diff_bg_of_t(t; t0=250)
    pd, A0, A1 = 250, 11, 3
    t < 250 ? Lee1997_bg_jet(1.0, A0) : Lee1997_bg_jet(1.0, A0 + A1 * sin(2 * pi * (t - t0) / pd))
end

# Topography

h0_topo = 0.0 # 2000.0       # Maximum height of the continental shelf (m)
W_shelf = WC # 100e3        # Shelfbreak width 
y_shelf = 0.0         # Central latitude of the shelf break

# Tanh profile: shallow on the Antarctic continent (south), deep ocean (north)
eta_b = @. (h0_topo / 2.0) * (1.0 - tanh((y' - y_shelf) / W_shelf))

# Scale to topographic PV
topo_PV = ones(Nx,1) * f0 .* eta_b ./ H2

# Calculate and print the maximum absolute gradients
max_grad_x = maximum(abs.(diff(topo_PV, dims=1) ./ dx))
max_grad_y = maximum(abs.(diff(topo_PV, dims=2) ./ dy))

println("Max Topographic PV Gradient (x-dir): " * @sprintf("%.3E", max_grad_x))   #, round(max_grad_x, digits=4))
println("Max Topographic PV Gradient (y-dir): " * @sprintf("%.3E", max_grad_y))      # , round(max_grad_y, digits=4))


################################################################################
# Localized Surface Wind Forcing (with internal edge taper)
################################################################################
τ0 = 0.0 # 0.15          # Maximum wind stress (N/m^2)
W_wind = 200e3   # Total width of the central wind forcing zone 

# 1. Calculate the continuous analytical wind curl over the entire domain
wind_curl = @. (τ0 * π) / (ρ0 * H1 * W_wind) * sin(2.0 * π * y / W_wind)

# 2. Build the taper array
taper = similar(y)
L_wind = W_wind / 2.0            # The absolute edge of the wind zone
r_taper_start = 0 #0.7 * L_wind     # Where the taper begins (e.g., 80% out)

for (i, yi) in enumerate(y)
    rt = abs(yi)
    
    if rt <= r_taper_start
        # Un-damped interior region
        taper[i] = 1.0
    elseif rt < L_wind
        # Smoothly squash the sine wave to zero as it approaches L_wind
        ξ = (rt - r_taper_start) / (L_wind - r_taper_start)   
        
        # Hanning window: 1.0 at r_taper_start, smoothly decaying to 0.0 at L_wind
        taper[i] = 0.5 * (1.0 + cos(pi * ξ))  
    else
        # Strictly zero everywhere outside the wind zone
        taper[i] = 0.0
    end
end

# 3. Apply the taper to squash the edges and zero out the exterior
wind_curl .*= taper

##################################################
## NO MOMENTUM FORCING
α_STJ_1D = 0.0

U_STJ_target = [0.0]


################################################################################
# 3. Output & Diagnostic Configuration
################################################################################
base_name = "restoreU$(fix_zonal_mean)_beta$(beta)_WC$(WC)_trans$(trans)_r$(r)_h0topo$(h0_topo)_Wshelf$(W_shelf)_U0$(U0)_tau0$(τ0)_alpha$(α)"
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
    topo_PV=topo_PV, wind_curl=wind_curl, t_start_diag=0, 
    output_every=250, fix_zonal_mean=fix_zonal_mean
)

include("./plot_Hov.jl")

