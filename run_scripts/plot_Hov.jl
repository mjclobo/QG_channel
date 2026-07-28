using JLD2, PyPlot, Statistics

PyPlot.pygui(true)


# 1. Define paths
base_name = "fixU$(fix_zonal_mean)_beta$(beta)_WC$(WC)_trans$(trans)_r$(r)_h0topo$(h0_topo)_Wshelf$(W_shelf)_U0$(U0)_tau0$(τ0)_alpha$(α)"
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
save_path = out_cfg.save_path
fig_path  = out_cfg.fig_path
diag_dir  = out_cfg.diag_dir


# Assuming the diagnostic suite saves each diag as a separate file, or one big file. 
# Adjust the filename depending on your DiagnosticSuite's save method.
t_diag = t0 + params.nt * dt
file_name = struct_to_string(params) * "_diags_t$t_diag.jld2"

#
#diag_file = joinpath(base_dir, "diagnostics", base_name, "HovmollerZonalFlow.jld2")

# 2. Load the data
data = load(joinpath(out_cfg.diag_dir, file_name))
println("Found keys in diagnostic file: ", keys(data))

# 3. Extract variables 
# (Adjust the string keys based on the printout above if they differ)
t_days = data["jld_data"]["HovmollerZonalFlow"]["times"][1:end-1] ./ (24 * 3600)  # Convert time from seconds to days
y_km = y ./ 1e3            # Convert y coordinates to kilometers
U1_hov = data["jld_data"]["HovmollerZonalFlow"]["U1"][:,1:end-1]            # Expected shape: (Ny, N_saved_timesteps)
U2_hov = data["jld_data"]["HovmollerZonalFlow"]["U2"][:,1:end-1]

# 4. Generate the Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=true)

# Layer 1
ax1_lim = maximum(abs.(U1_hov))
pc1 = ax[1].pcolormesh(t_days, y_km, U1_hov, cmap=PyPlot.cm.RdBu_r, vmin=-ax1_lim, vmax=ax1_lim, shading="nearest")
ax[1].set_title("Layer 1 Zonal Mean Flow (U₁)")
ax[1].set_xlabel("Time (days)")
ax[1].set_ylabel("y (km)")
plt.colorbar(pc1, ax=ax[1], label="m/s")

# Layer 2
ax2_lim = maximum(abs.(U2_hov))
pc2 = ax[2].pcolormesh(t_days, y_km, U2_hov, cmap=PyPlot.cm.RdBu_r, vmin=-ax2_lim, vmax=ax2_lim, shading="nearest")
ax[2].set_title("Layer 2 Zonal Mean Flow (U₂)")
ax[2].set_xlabel("Time (days)")
plt.colorbar(pc2, ax=ax[2], label="m/s")

fig.tight_layout()
plt.show()

# Optional: Save the figure
 save_path = joinpath(base_dir, "anim", base_name, "Hovmoller_final.png")
 PyPlot.savefig(save_path, dpi=300)
