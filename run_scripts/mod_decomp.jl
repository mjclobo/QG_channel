################################################################################
# Load packages & Source
################################################################################
using FFTW, LinearAlgebra, Statistics, Dates, JLD2, PyPlot, Printf
using Random: seed!
using SparseArrays, Base.Threads, LoopVectorization, Parameters, KernelAbstractions

################################################################################
# 1. Core Parameters & Geometry
################################################################################
Nx, Ny, Nz = 64, 128, 2
Lx, Ly = 64, 128  

dx = Lx / (Nx - 1) # Assuming standard grid spacing
dy = Ly / (Ny - 1)
x = collect(range(0, Lx, length=Nx))
y = collect(range(-Ly/2, Ly/2, length=Ny))

# Physics
beta = 0.0 # 0.25
f0, g, ρ0 = 1.0, 1.0, 1.0
H1, H2 = 2.0, 2.0
Δρ = 1.0  
U0 = 1.0  

gprime = g * Δρ / ρ0
F1 = 2 * f0^2 / (gprime * H1)
F2 = 2 * f0^2 / (gprime * H2)

# Timestepping
cfl = 0.04  
dt = cfl * minimum([dx, dy]) / U0      
ndays = 500
nt = round(Int, ndays/dt)                              

# Damping
N_steps = 5.0
ν = 10 * (dx^4) / (N_steps * dt * (2π)^4)
r = 0.1         
α = 1.0 / 10.0   

# Jet Parameters
trans = 1.0
WC = 35.0 

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
ψ1_bg_1D, U_bg, zone_start_ind, zone_end_ind = Lee1997_bg_jet(U0, WC; σ=5.0)

# Expand the 1D profile to an Nx by Ny matrix
ψ1_bg = repeat(ψ1_bg_1D', Nx, 1) 
ψ2_bg = zeros(Nx, Ny)
ψ_diff_bg = ψ1_bg .- ψ2_bg

bg_of_t = true
function ψ_diff_bg_of_t(t; t0=250)
    pd, A0, A1 = 250, 11, 3
    t < 250 ? Lee1997_bg_jet(1.0, A0) : Lee1997_bg_jet(1.0, A0 + A1 * sin(2 * pi * (t - t0) / pd))
end

# Topography
H0 = 0. # 1.0        
x0, y0 = Lx / 2, 0.0      
σx, σy = 5.0, 5.0        
eta_b = @. H0 * exp(-((x - x0)^2 / (2 * σx^2) + (y' - y0)^2 / (2 * σy^2)))
topo_PV = eta_b

# Calculate and print the maximum absolute gradients
max_grad_x = maximum(abs.(diff(topo_PV, dims=1) ./ dx))
max_grad_y = maximum(abs.(diff(topo_PV, dims=2) ./ dy))

println("Max Topographic PV Gradient (x-dir): ", round(max_grad_x, digits=4))
println("Max Topographic PV Gradient (y-dir): ", round(max_grad_y, digits=4))

################################################################################
# 3. Output & Diagnostic Configuration
################################################################################
base_name = "WC_init_beta$(beta)_WC$(WC)_trans$(trans)_r$(r)"
base_dir = "/home/matt/Desktop/research/QG/2LQG/HeldLobo_proj/QG_channel_output"

# Define frequencies as standalone variables first
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
# 4. State Initialization
################################################################################
seed!(1234)

# Use the 1D versions to compute the 1D background PV!
ψ2_bg_1D = zeros(Ny)
q1_bg, q2_bg = compute_qg_pv_bar(ψ1_bg_1D, ψ2_bg_1D)

q1_prime = 1.0 * randn(Nx, Ny)   
q2_prime = 1.0 * randn(Nx, Ny)   
q1_prime[:, 1] .= 0.0; q1_prime[:, end] .= 0.0;
q2_prime[:, 1] .= 0.0; q2_prime[:, end] .= 0.0;

# Build the unified structs (Note: bar arrays are 1D Vectors of length Ny!)
state = QGState(
    q1_prime, q2_prime, zeros(Ny), zeros(Ny), 
    zeros(Nx, Ny), zeros(Nx, Ny), zeros(Ny), zeros(Ny)
)
ops = QGOperators(solver2L, A_lu, rhs_pa, ψ_vec, L1D, L2D)
model = QGModel(state, ops)

################################################################################
# 5. Diagnostic Suite Setup
################################################################################
suite = DiagnosticSuite()
add!(suite, :Pseudomomentum, PseudomomentumDiag(nt, dt, diag_every, Ny))
add!(suite, :ZonalMeanEnergy, ZonalMeanEnergyDiag(nt, dt, diag_every, Ny))
add!(suite, :ScalarEnergy, ScalarEnergyDiag(nt, dt, diag_every, zone_start_ind, zone_end_ind))

################################################################################
# 6. Run Model
################################################################################
t0 = 0.0

# The beautifully clean function call!
run_model_decomp(model, suite, out_cfg, ψ_diff_bg, U_bg, t0, params; topo_PV=topo_PV, t_start_diag=250)




# now you can run L97_WC4_SS.jl to calculate steady-state turbulent statistics

# U_bg_all = zeros(11, Ny)

# fig, ax = plt.subplots(1, 1, figsize=(10,5))

# for (i, t) in enumerate(range(0, 1, 11))
#     ψ1_bg, U_bg_all[i,:], zone_start_ind, zone_end_ind = blended_transport_jet(y; T=0, W=15, trans=t)

#     ax.plot(y, U_bg_all[i,:])

# end
