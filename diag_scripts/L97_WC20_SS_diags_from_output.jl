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
H1 = 5.0 # 2000.0
H2 = 5.0 # 2000.0
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
nt = 20000                              # number of time steps

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

WC = 20  # Width of boundary where background flow decays to zero (max of 0.5)

ψ_diff_bg, U_bg = Lee1997_bg_jet(U0, WC)

ψ_diff_bg = ψ_diff_bg .* ones(Nx, Ny)

################################################################################
# Define paths where saved data is
################################################################################

save_path = "/home/matt/Desktop/research/QG/QG_channel_output/data/L97_WC20_SS/"
y_width = 0.6   # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain

fig_path = "/home/matt/Desktop/research/QG/QG_channel_output/diags/L97_WC20_SS/"

################################################################################
# Damping (biharmonic viscosity, linear friction, and thermal damping)
################################################################################
ν = 1e-3 #  0.01 * dx^4 / dt # 1e6          # Hyperviscosity (m⁴/s)

r = 0.05         # Ekman friction (1/s)
α = 30^-1        # Thermal damping (1/s)

################################################################################
# Define model params struct
################################################################################

include("../define_vars.jl")
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1, H2], ρ0, Δρ, ν, r, α, U0, WC)

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

fig, ax = plt.subplots(1, 2, figsize=(10,5))

ax[1].pcolormesh(-c, y[save_ind_start:save_ind_end], heat_flux', vmin=-maximum(abs.(heat_flux)), vmax=maximum(abs.(heat_flux)), cmap=PyPlot.cm.bwr)
ax[1].plot(Ubar[:,1], y[save_ind_start:save_ind_end])
ax[1].plot(Ubar[:,3], y[save_ind_start:save_ind_end])
plt.grid()


ax[2].pcolormesh(-c, y[save_ind_start:save_ind_end], vort_flux', vmin=-maximum(abs.(vort_flux)), vmax=maximum(abs.(vort_flux)), cmap=PyPlot.cm.bwr)
ax[2].plot(Ubar[:,1], y[save_ind_start:save_ind_end])
ax[2].plot(Ubar[:,3], y[save_ind_start:save_ind_end])
plt.grid()



sum_savename = @sprintf("%s.png", joinpath(fig_path, "heat_vort_flux"))
PyPlot.savefig(sum_savename)

