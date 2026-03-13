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
using ImageFiltering, ImageCore
using Parameters

using KernelAbstractions
# using CUDA  # only for GPU


################################################################################
# Geometry
################################################################################

Nx = 256
Ny = 256
Lx = 25000e3 # 20 
Ly = 25000e3 # 20 
dx = Lx / Nx
dy = Ly / Ny

# Do not change these vvv
x = collect(-(Lx-dx)/2:dx:(Lx-dx)/2)   # collect(0:dx:Lx-dx)
y = collect(-(Ly-dy)/2:dy:(Ly-dy)/2)   # collect(0:dy:Ly-dy)

################################################################################
# Model params
################################################################################

beta = 0.0  # 0.25 #

# Do not change these vvv
f0 =  1.125e-4 # 1.0 #
g = 9.81 # 1.0 # 
Ld = 1000.0e3  # sqrt(g*H)/f0
H = (Ld * f0)^2 / g

Nz=1

U0 = 0.

# η = 0.05 * cos.(2*pi*x./10) .* sin.(2*pi*y ./ 10)' # zeros(Nx, Ny)

################################################################################
# Timestepping params
################################################################################

cfl = 0.5      # nominal CFL

# dt = cfl * minimum([dx, dy]) / U0       # time step
dt = cfl * dx / 30 # target is 30 m/s jet maximum; 0.005
nt = 15000                              # number of time steps

timestep_method = "RK4" # "RK4_int"     # options are: RK4, RK4_int

################################################################################
# Load source code
################################################################################

src_dir = "/home/matt/Desktop/research/QG/QG_channel/src/"
src_files = readdir(src_dir)
for file in src_files include(src_dir*file) end

################################################################################
# Define paths for saving streamfunction files and figures; and frequency of output
################################################################################

# this saves meridional bands (full zonal extent) of i) ψ1, ii) ψ2, and iii) t
save_bool = true
save_path = "/home/matt/Desktop/research/QG/QG_channel_output/data/BT_model/"
y_width = 0.75  # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain
save_every = round(Int,nt/20)      # period of save frequency


# this saves full streamfunction field (and time) at end of simulation run, in case you'd like to use the field as initial conditions later on
# directory is defined by save_path (above)
save_last = true

# this plots panels at fig_path; the plot function (defined in output_fcns.jl) can be modified to be whatever you want to see
fig_path = "/home/matt/Desktop/research/QG/QG_channel_output/anim/BT_model/"
plot_every = round(Int,nt/200)      # period of plot output frequency


################################################################################
# Damping (biharmonic viscosity, linear friction, and thermal damping)
################################################################################
ν = 0.01 * dx^4 / dt # 1e6          # Hyperviscosity (m⁴/s)  

r = 0.0 # (30 * 24 * 3600)^-1  # 0.05         # Ekman friction (1/s)  

################################################################################
# Set initial conditions
################################################################################
include("../define_vars.jl")
params = BTParams(Nx, Ny, nt, Lx, Ly, dt, beta, Ld, ν, r, U0)

kx = reshape(fftfreq(Nx, 1/Lx*Nx), (Nx,1))
ky = reshape(fftfreq(Ny, 1/Ly*Ny), (1,Ny))
# kxr = reshape(rfftfreq(Nx, 2π/Lx*Nx), (Int(Nx/2 + 1),1))

k2D = kx.^2 .+ ky.^2

# seed!(1234)
# qh_init = 0.0 * randn(Complex{Float64}, (Nx, Ny))    #  zeros((Nx, Ny))  #
# @. qh_init = ifelse(k2D < 5  * 2π/Lx, 0, qh_init)
# @. qh_init = ifelse(k2D > 30 * 2π/Lx, 0, qh_init)
# @. qh_init[1, :] = 0    # remove any power from zonal wavenumber k=0

t0 = 0    # initial timestamp (in seconds)

center_x = Lx/2; center_y = Ly/2;
radius = 5*Ld

q0 = f0 .* ones(Nx, Ny)
for i in range(1,Nx)
    for j in range(1, Ny)
        if sqrt(x[i]^2 + y[j]^2) < radius
            q0[i,j] += 1.4192 * f0 # 1.0
        end
    end
end

nf = 8*pi / (1.192*f0)/3600/24   # natural period of oscillations

# # Define smoothing kernel
kernel = Kernel.gaussian(3) # '3' is the standard deviation for the blur

# Apply the filter
q0 = imfilter(q0, kernel)
q0 .-= mean(q0)

qh_init = fft(q0);




function def_psi(qh)
    # ψf = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

    # ψfh = fft(ψf)

    # topo_forc = @. -(-k2D - Ld^-2) * ψfh

    # qh .+= topo_forc

    # q = real.(ifft(qh))

    L_inv = -(k2D .+ Ld^-2).^-1

    L_inv[1,1] = 0.

    ψh = L_inv .* qh

    ψ = real.(ifft(ψh))

    return ψ, ψh
end

# ################################################################################
# # Defining ``wavemaker'' forcing by time-dependent topography
# ################################################################################
function wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)
    return @. ψ0 * sin(2 * pi * t / τ) * exp(-((x - x0)^2 / (2 * δx^2) + (y' - y0)^2 / (2 * δy^2)))
end

# parameters from Nakamura and Plumb 1994, p. 2033
x0 = -radius
y0 = 0

δx = 0.3 * radius
δy = 0.2 * radius

τ = 3.73 * 24 * 3600 # 20.0

ψ_init, ψh_init = def_psi(qh_init)

ψ0 = 0.15 * maximum(ψ_init)

t=0

# ψf_init = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

# ψfh_init = fft(ψf_init)

# topo_forc_init = @. -(-k2D - Ld^-2) * ψfh_init

# qh_init .+= topo_forc_init

# t = τ/4
# plt.pcolormesh(wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)')
# plt.colorbar()

################################################################################
# Run model from initial conditions
################################################################################
# params = init_solver(qh_init, kx, ky, k2D, Ld,ψ0, x, y, x0, y0, δx, δy)

# run_BT_model(qh_init, t0, params)

# run_BT_model_FD(q0, t0, params)


params = init_params_BT_FD(Nx, Ny, dx, dy, r, f0, L_re_FD_inv, L_re_FD, Lap_op; backend=CPU())   # use CUDABackend() in place of CPU() for GPU

run_BT_model_FD_KA(q0, params; nt=10_000, dt, output_every=500)



# PyPlot.pygui(true)

# ψ, ψh = def_psi(qh_init)

# plt.pcolormesh(real.(ifft(im .* kx .* ψh)))

# plt.colorbar()


