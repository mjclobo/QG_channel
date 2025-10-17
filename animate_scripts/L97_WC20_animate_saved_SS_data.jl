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
using LaTeXStrings

# still need to define all model params to generate saved file names

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
# Define paths for saving streamfunction files and figures; and frequency of output
################################################################################

save_path = "/home/matt/Desktop/research/QG/QG_channel/data/L97_WC20_SS/"
y_width = 0.6   # meridional width of domain that is saved; max of 1 will save whole meridional extent of domain


fig_path = "/home/matt/Desktop/research/QG/QG_channel/anim/L97_WC20_anim_output/"
plotname = "snapshots"

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
# Define model params
################################################################################
include("../define_vars.jl")
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1, H2], ρ0, Δρ, ν, r, α, U0, WC)

# ################################################################################
# # Generate plots for animation; 
# ################################################################################
save_ind_start = floor(Int, Ny * y_width / 2)
save_ind_end   = floor(Int, Ny * (1 - y_width / 2))
Ny_saved = save_ind_end - save_ind_start + 1

# define Laplacian for meridional band that we saved
L2D_yband = laplacian_operator_neumann_x(Nx, Ny_saved, dx, dy)

t_array = define_t_of_saved_files(Ny, save_path)

file_start, file_end = extract_time(readdir(save_path)[1]; file_strings=true)

PyPlot.pygui(false)

ymin_ind = save_ind_start + 2

ymax_ind = save_ind_end - 2

# finding time-mean of all terms as well
v1_bar_star_tm = zeros(Ny_saved)
v1η_bar_tm = zeros(Ny_saved)
v1ζ1_bar_tm = zeros(Ny_saved)

v2_bar_star_tm = zeros(Ny_saved)
v2η_bar_tm = zeros(Ny_saved)
v2ζ2_bar_tm = zeros(Ny_saved)
drag_tm = zeros(Ny_saved)

global mean_count = 0

##

# i=1
# t=t_array[i]
for (i, t) in enumerate(t_array[1:4:end])
    savedata = load(save_path * file_start * string(t) * file_end)

    ψ1_snap = savedata["jld_data"]["ψ1"]
    ψ2_snap = savedata["jld_data"]["ψ2"]

    ψ1_prime = ψ1_snap .- mean(ψ1_snap, dims=1)
    ψ2_prime = ψ2_snap .- mean(ψ2_snap, dims=1)

    # define PV
    q1_snap, q2_snap = compute_qg_pv(ψ1_snap, ψ2_snap; lap_op=L2D_yband)

    q1_bar = mean(q1_snap, dims=1)
    q2_bar = mean(q2_snap, dims=1)

    # define velocities
    u1_snap, v1_snap = u_from_psi(ψ1_snap)
    u2_snap, v2_snap = u_from_psi(ψ2_snap)

    u1_bar = mean(u1_snap, dims=1)
    u2_bar = mean(u2_snap, dims=1)

    v1_bar, v2_bar = diagnose_zonal_mean_meridional_velocity(ψ1_snap, ψ2_snap, dx, dy, beta, f0, gprime, H1, H2, L2D_yband)
    # v1_bar, v2_bar = diagnose_meridional_velocity_from_omega_nonperiodic_y(ψ1_snap, ψ2_snap, dx, dy)
        
    u1_prime = u1_snap .- u1_bar
    v1_prime = v1_snap # .- v1_bar

    u2_prime = u2_snap .- u2_bar
    v2_prime = v2_snap # .- v2_bar

    # define vorticities
    ζ1_snap = L2D_yband(ψ1_snap)
    ζ2_snap = L2D_yband(ψ2_snap)

    ζ1_prime = ζ1_snap .- mean(ζ1_snap, dims=1)
    ζ2_prime = ζ2_snap .- mean(ζ2_snap, dims=1)

    # define thicknesses
    η_snap = f0 * (ψ2_snap .- ψ1_snap) / gprime

    η_bar = mean(η_snap, dims=1)

    η_prime = η_snap .- η_bar

    # define heat/thickness fluxes
    v1η_bar = mean(v1_prime .* η_prime, dims=1)
    v2η_bar = mean(v2_prime .* (-η_prime), dims=1)

    # define residual velocity
    v1_bar_star = v1_bar .+ v1η_bar./ H1
    v2_bar_star = v2_bar .+ v2η_bar ./ H2    

    # define vorticity fluxes
    v1ζ1_bar = mean(v1_prime .* ζ1_prime, dims=1)
    v2ζ2_bar = mean(v2_prime .* ζ2_prime, dims=1)
    
    # define bottom drag
    drag = -r * u2_bar

    # plot
    fig, ax = plt.subplots(3, 4, figsize=(16,12), height_ratios=[1.0, 0.3, 1.0], width_ratios=[1.0, 1.0, 0.4, 0.4])

    # pv in left column
    ax1lim = maximum(abs.(q1_snap .- q1_bar))
    ax[1].pcolormesh(x, y[save_ind_start:save_ind_end], (q1_snap .- q1_bar)', cmap=PyPlot.cm.bwr, vmin=-ax1lim, vmax=ax1lim)
    ax[1].set_title(L"q1")

    ax3lim = maximum(abs.(q2_snap .- q2_bar))
    ax[3].pcolormesh(x, y[save_ind_start:save_ind_end], (q2_snap .- q2_bar)', cmap=PyPlot.cm.bwr, vmin=-ax3lim, vmax=ax3lim)
    ax[3].set_title(L"q2")

    for axn in [ax[1], ax[3]]
        axn.set_xlabel(L"x")
        axn.set_ylabel(L"y")
        axn.set_ylim(y[ymin_ind], y[ymax_ind])
    end

    ax[2].axis("off")

    # streamfunction in middle left column (with interface height in center row)
    ax4lim = maximum(abs.(ψ1_prime))
    ax[4].contourf(x, y[save_ind_start:save_ind_end], (ψ1_prime)', cmap=PyPlot.cm.PiYG, vmin=-ax4lim, vmax=ax4lim, levels=7)
    ax[4].set_title(L"ψ1")

    ax6lim = maximum(abs.(ψ2_prime))
    ax[6].contourf(x, y[save_ind_start:save_ind_end], (ψ2_prime)', cmap=PyPlot.cm.PiYG, vmin=-ax6lim, vmax=ax6lim, levels=7)
    ax[6].set_title(L"ψ2")

    for axn in [ax[4], ax[6]]
        axn.set_xlabel(L"x")
        axn.set_ylabel(L"y")
        axn.set_ylim(y[ymin_ind], y[ymax_ind])
    end

    ax[5].plot(y[save_ind_start:save_ind_end], η_bar', "r--", label=L"\overline{\eta}")
    ax[5].plot(y[save_ind_start:save_ind_end], η_bar' .+  5 * η_prime[round(Int, Nx/2), :], "k-", label=L"\overline{\eta} + 5 \, \eta^{\prime}")
    ax[5].set_xlim(y[save_ind_start], y[save_ind_end])

    ax[5].legend(loc="lower right")
    ax[5].set_ylabel(L"\eta \, \mathrm{[m]}")
    ax[5].set_xlabel(L"y")

    # zonal mean flow and restoring profile in middle right column
    ax[7].plot(U_bg', y, "r--")
    ax[7].plot(u1_bar', y[save_ind_start:save_ind_end], "k-")
    ax[7].set_title(L"\overline{u}_{1}")

    ax[9].plot(zeros(size(U_bg)), y, "r--")
    ax[9].plot(u2_bar', y[save_ind_start:save_ind_end], "k-")
    ax[9].set_title(L"\overline{u}_{2}")

    for axn in [ax[7], ax[9]]
        axn.set_ylabel(L"y")
        axn.set_ylim(y[ymin_ind], y[ymax_ind])
        axn.set_xlim(-1, 3)
    end

    ax[8].axis("off")

    # time-evolution equation of zonal mean flow in rightmost columnn
    ax[10].plot(f0 .* v1_bar_star', y[save_ind_start:save_ind_end], alpha=0.5, label=L"f_{0} \, \overline{v_{1}}^{*}")
    ax[10]. plot(-(f0 * v1η_bar ./ H1)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"- f_{0} \, \overline{v_{1}^{\prime} \, H_{1}^{\prime} } / H_{o,1}")
    ax[10].plot(v1ζ1_bar', y[save_ind_start:save_ind_end], alpha=0.5, label=L"\overline{v_{1}^{\prime} \, \zeta_{1}^{\prime} }")

    ax[10].plot((v1ζ1_bar .- (f0 * v1η_bar ./ H1) .+ f0 .* v1_bar_star)', y[save_ind_start:save_ind_end], "k-", linewidth=2, label=L"\partial_{t} \, \overline{u}_{1}")

    ax[12].plot(f0 .* v2_bar_star', y[save_ind_start:save_ind_end], alpha=0.5, label=L"f_{0} \, \overline{v_{2}}^{*}")
    ax[12]. plot(-(f0 * v2η_bar ./ H2)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"-f_{0} \, \overline{v_{2}^{\prime} \, H_{2}^{\prime} } / H_{o,2}")
    ax[12].plot(v2ζ2_bar', y[save_ind_start:save_ind_end], alpha=0.5, label=L"\overline{v_{2}^{\prime} \, \zeta_{2}^{\prime} }")
    ax[12].plot(drag', y[save_ind_start:save_ind_end], alpha=0.5, label=L"- r \overline{u_{2}}")

    ax[12].plot((v2ζ2_bar .- (f0 * v2η_bar ./ H2) .+ f0 .* v2_bar_star .+ drag)', y[save_ind_start:save_ind_end], "k-", linewidth=2, label=L"\partial_{t} \, \overline{u}_{2}")

    for axn in [ax[10], ax[12]]
        axn.set_ylabel(L"y")
        axn.set_ylim(y[ymin_ind], y[ymax_ind])
        axn.set_xlim(-0.25, 0.25)
    end

    ax[10].legend(loc="upper right", bbox_to_anchor=[1.875, 0.5])
    ax[12].legend(loc="upper right", bbox_to_anchor=[1.875, 0.5])

    ax[11].axis("off")

    # save figure and close plot
    if mod(i, 10)==0
        println("Saving panel $i")
    end

    savename = @sprintf("%s_%04d.png", joinpath(fig_path, plotname), i)
    PyPlot.savefig(savename)

    # calculating time means
    v1_bar_star_tm .+= v1_bar_star'
    v1η_bar_tm .+= v1η_bar'
    v1ζ1_bar_tm .+= v1ζ1_bar'

    v2_bar_star_tm .+= v2_bar_star'
    v2η_bar_tm .+= v2η_bar'
    v2ζ2_bar_tm .+= v2ζ2_bar'
    drag_tm .+= drag'

    global mean_count+=1

    PyPlot.close()
end


PyPlot.pygui(true)

fig, ax = plt.subplots(1, 2, figsize=(6, 12))

ax[1].plot((f0 .* v1_bar_star_tm ./ mean_count)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"f_{0} \, \overline{v_{1}}^{*}")
ax[1]. plot(-(f0 * v1η_bar_tm ./ (mean_count * H1))', y[save_ind_start:save_ind_end], alpha=0.5, label=L"- f_{0} \, \overline{v_{1}^{\prime} \, H_{1}^{\prime} } / H_{o,1}")
ax[1].plot((v1ζ1_bar_tm ./ mean_count)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"\overline{v_{1}^{\prime} \, \zeta_{1}^{\prime} }")

ax[1].plot(((v1ζ1_bar_tm .- (f0 * v1η_bar_tm ./ H1) .+ f0 .* v1_bar_star_tm) ./ mean_count)', y[save_ind_start:save_ind_end], "k-", linewidth=2, label=L"\partial_{t} \, \overline{u}_{1}")


ax[2].plot((f0 .* v2_bar_star_tm ./ mean_count)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"f_{0} \, \overline{v_{2}}^{*}")
ax[2]. plot(-(f0 * v2η_bar_tm ./ (mean_count * H2))', y[save_ind_start:save_ind_end], alpha=0.5, label=L"-f_{0} \, \overline{v_{2}^{\prime} \, H_{2}^{\prime} } / H_{o,2}")
ax[2].plot((v2ζ2_bar_tm ./ mean_count)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"\overline{v_{2}^{\prime} \, \zeta_{2}^{\prime} }")
ax[2].plot((drag_tm ./ mean_count)', y[save_ind_start:save_ind_end], alpha=0.5, label=L"- r \overline{u_{2}}")

ax[2].plot(((v2ζ2_bar_tm .- (f0 * v2η_bar_tm ./ H2) .+ f0 .* v2_bar_star_tm .+ drag_tm) ./ mean_count)', y[save_ind_start:save_ind_end], "k-", linewidth=2, label=L"\partial_{t} \, \overline{u}_{2}")


##
for axn in [ax[1], ax[2]]
    axn.set_ylabel(L"y")
    axn.set_ylim(y[ymin_ind], y[ymax_ind])
    axn.set_xlim(-0.1, 0.1)
    axn.legend(loc="upper left", bbox_to_anchor=[0.25,1.05])
end

sum_savename = @sprintf("%s.png", joinpath(fig_path, "mom_budget"))
PyPlot.savefig(sum_savename)
