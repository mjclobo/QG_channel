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

using SparseArrays
using Base.Threads
using LoopVectorization

################################################################################
# Geometry
################################################################################

Nx = 128
Ny = 128*2
Lx = 90/2 #   500e3 * Nx
Ly = 90 #   500e3 * Ny
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
H1 = 5 # 2000.0
H2 = 5 # 2000.0
ρ0 = 1.0
Δρ = 0.5  # 0.75

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
nt = 30000                              # number of time steps

timestep_method = "RK4" # "RK4_int"     # options are: RK4, RK4_int

################################################################################
# Load source code
################################################################################

src_dir = "/home/matt/Desktop/research/QG/channel_model/src/"
src_files = readdir(src_dir)
for file in src_files include(src_dir*file) end

################################################################################
# Define background flow profile
################################################################################

WC = 10  # Width of boundary where background flow decays to zero (max of 0.5)

ψ_diff_bg, U_bg = Lee1997_bg_jet(U0, WC)

ψ_diff_bg = ψ_diff_bg .* ones(Nx, Ny)

################################################################################
# Define paths for saving streamfunction files and figures; and frequency of output
################################################################################

# this saves meridional bands (full zonal extent) of i) ψ1, ii) ψ2, and iii) t
save_bool = true
save_path = "/home/matt/Desktop/research/QG/channel_model/data/L97_WC10_SS/"
y_width = 0.6   # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain
save_every = round(Int,nt/1000)      # period of save frequency


# this saves full streamfunction field (and time) at end of simulation run, in case you'd like to use the field as initial conditions later on
# directory is defined by save_path (above)
save_last = false


# this plots panels at fig_path; the plot function (defined in output_fcns.jl) can be modified to be whatever you want to see
plot_bool = true
fig_path = "/home/matt/Desktop/research/QG/channel_model/anim/L97_WC10_SS/"
plot_every = round(Int,nt/500)      # period of plot output frequency


################################################################################
# Damping (biharmonic viscosity, linear friction, and thermal damping)
################################################################################
ν = 1e-3 #  0.01 * dx^4 / dt # 1e6          # Hyperviscosity (m⁴/s)

# # an alternate method for setting ν
# nv = 2 # 2nd order hyperviscosity
# kmax = 2π * Nx / 2 / Lx
# nutune = 2.0                # 0.1 is good for dx=0.2 (32 grid points for L=2π)
# ν = nutune*dx/(dt*kmax^(2*nv)) #        % del^(2*hv) hypervisc

r = 0.05         # Ekman friction (1/s)
α = 30^-1        # Thermal damping (1/s)

################################################################################
# Set initial conditions using a restart file
################################################################################
restart_path = "/home/matt/Desktop/research/QG/channel_model/data/L97_WC10_init/"

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

# run_model(q1, q2, t0, params)

# ################################################################################
# # Diagnose latitude-time frequency spectra of eddy fluxes of heat and vorticity
# ################################################################################
cmax = 3.0          # maximum phase speed considered
ncbins = round(Int, 128)             # number of phase speed bins

spec_params = make_spectral_params(params, dx, dy, cmax, ncbins)

# construct Nx x Ny_saved x nt arrays of ψ1 and ψ2
save_ind_start = floor(Int, Ny * y_width / 2)
save_ind_end   = floor(Int, Ny * (1 - y_width / 2))
Ny_saved = save_ind_end - save_ind_start + 1

t_array = define_t_of_saved_files(Ny, save_path)
ψ1_of_t, ψ2_of_t = construct_psi_of_t(t_array, Ny_saved, save_path)


# calculate spectra of eddy fluxes
heat_flux, vort_flux, c, Ubar = compute_phase_speed_spectra(ψ1_of_t, ψ2_of_t, spec_params) # Ubar is U1, ∂yU1, U2, ∂yU2


# # alternately, use moving blocks
block_len = round(Int, length(t_array)/2)
block_stride = round(Int, 0.5 * block_len)
heat_flux, vort_flux, c = compute_flux_spectra_block(ψ1_of_t, ψ2_of_t, params, dx, dy, spec_params, block_len, block_stride)

# heat_flux = copy(F_heat)
# vort_flux = copy(F_vort)

plt.pcolormesh(-c, y[save_ind_start:save_ind_end], heat_flux', vmin=-maximum(abs.(heat_flux)), vmax=maximum(abs.(heat_flux)), cmap=PyPlot.cm.bwr)
plt.plot(Ubar[:,1], y[save_ind_start:save_ind_end])
plt.plot(Ubar[:,3], y[save_ind_start:save_ind_end])
plt.grid()


plt.figure()
plt.pcolormesh(-c, y[save_ind_start:save_ind_end], vort_flux', vmin=-maximum(abs.(vort_flux)), vmax=maximum(abs.(vort_flux)), cmap=PyPlot.cm.bwr)
plt.plot(Ubar[:,1], y[save_ind_start:save_ind_end])
plt.plot(Ubar[:,3], y[save_ind_start:save_ind_end])
plt.grid()

