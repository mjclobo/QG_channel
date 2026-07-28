
################################################################################
# Data write functions
################################################################################

function struct_to_string(s::T) where T
    field_strings = String[]
    for field_name in fieldnames(T)
        field_value = getfield(s, field_name)
        if startswith(string(field_name), "Δρ")
            push!(field_strings, "drho$field_value")
        elseif startswith(string(field_name), "H")
            push!(field_strings, "H1"*string(field_value[1]))
            push!(field_strings, "H2"*string(field_value[2]))
        elseif startswith(string(field_name), "ψ_diff")
            # do nothing
        else
            push!(field_strings, "$field_name"*string(round(field_value, digits=3)))
        end
    end
    return join(field_strings, "_")
end

function save_streamfunction(dir, ψ1, ψ2, t, params)
    file_name = struct_to_string(params) * "_t$t.jld2"
    
    # No need to wrap in Array() if they are standard Matrix{Float64}
    jld_data = Dict("ψ1" => ψ1, "ψ2" => ψ2, "t" => t)
    jldsave(joinpath(dir, file_name); jld_data)

    println("Saved streamfunction to $file_name")
end

function save_streamfunction(dir, ψ, t, params)
    file_name = struct_to_string(params) * "_t$t.jld2"
    
    jld_data = Dict("ψ" => ψ, "t" => t)
    jldsave(joinpath(dir, file_name); jld_data)

    println("Saved streamfunction to $file_name")
end


function save_basic_anim_panel(fig_path, ell, model::QGModel, U_bg, t, ψ_bg, q1_bg, q2_bg, fix_zonal_mean)
    plotname = "snapshots"
    s = model.state

    # 1. Reconstruct TOTAL fields based on the framework type
    if fix_zonal_mean
        ψ1 = ψ_bg .+ s.ψ1_prime
        ψ2 = s.ψ2_prime             # ψ2_bg is zero
        q1 = q1_bg .+ s.q1_prime
        q2 = q2_bg .+ s.q2_prime
    else
        ψ1 = s.ψ1_bar' .+ s.ψ1_prime
        ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        q1 = s.q1_bar' .+ s.q1_prime
        q2 = s.q2_bar' .+ s.q2_prime
    end

    # Pre-calculate zonal means and anomalies
    ψ1_zm = mean(ψ1, dims=1)
    ψ2_zm = mean(ψ2, dims=1)
    q1_zm = mean(q1, dims=1)
    q2_zm = mean(q2, dims=1)

    ψ1_anom = ψ1 .- ψ1_zm
    ψ2_anom = ψ2 .- ψ2_zm
    q1_anom = q1 .- q1_zm
    q2_anom = q2 .- q2_zm

    fig, ax = plt.subplots(2, 3, figsize=(14, 10), width_ratios=[1., 1., 0.4])

    ax2lim = maximum(abs.(ψ1_anom))
    pc3 = ax[3].contourf(x, y, ψ1_anom', cmap=PyPlot.cm.PiYG, vmin=-ax2lim, vmax=ax2lim, levels=7)
    ax[3].set_title("ψ1  at t = " * string(round(Int, t/3600/24)) * " days")
    plt.colorbar(pc3)

    ax4lim = maximum(abs.(ψ2_anom))
    pc4 = ax[4].contourf(x, y, ψ2_anom', cmap=PyPlot.cm.PiYG, vmin=-ax4lim, vmax=ax4lim, levels=7)
    ax[4].set_title("ψ2")
    plt.colorbar(pc4)

    ax1lim = maximum(abs.(q1_anom))
    pc = ax[1].pcolormesh(x, y, q1_anom', cmap=PyPlot.cm.bwr, vmin=-ax1lim, vmax=ax1lim)
    ax[1].set_title("q1 Anomaly")
    plt.colorbar(pc)

    ax3lim = maximum(abs.(q2_anom))
    pc2 = ax[2].pcolormesh(x, y, q2_anom', cmap=PyPlot.cm.bwr, vmin=-ax3lim, vmax=ax3lim)
    ax[2].set_title("q2 Anomaly")
    plt.colorbar(pc2)

    for axn in ax[1:4]
        axn.set_xlabel("x")
        axn.set_ylabel("y")
    end

    # 2. Calculate true total velocities including the background shear
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

    ax[5].plot(U_bg', y, "r--", label="therm. target")
    if length(U_STJ_target)>1
    	ax[5].plot(U_STJ_target', y, "b--", label="mom. target")
    end
    ax[5].plot(mean(u1, dims=1)', y, "k-", label="Actual Zonal Mean")
    ax[5].set_title("U_1 (zonal avg.)")
    #ax[5].set_xlim(-0.5*maximum(U_bg), 2.5*maximum(U_bg))

    ax[6].plot(zeros(size(U_bg)), y, "r--")
    ax[6].plot(mean(u2, dims=1)', y, "k-")
    ax[6].set_title("U_2 (zonal avg.)")
    #ax[6].set_xlim(-0.5*maximum(U_bg), 2.5*maximum(U_bg))

    for axn in ax[5:6]
        axn.set_xlabel("x")
        axn.set_ylim(y[1], y[end])
    end

    savename = @sprintf("%s_%04d.png", joinpath(fig_path, plotname), ell)
    PyPlot.savefig(savename)
    PyPlot.close()
end



function save_growth_plot(fig_path, ell, model::QGModel, U_bg, dt, nt, KE1_hist, KE2_hist, q1_bg, q2_bg, ψ_diff_bg)
    s = model.state
    
    # Calculate total fields once
    ψ1 = s.ψ1_bar' .+ s.ψ1_prime
    ψ2 = s.ψ2_bar' .+ s.ψ2_prime
    q1 = s.q1_bar' .+ s.q1_prime
    q2 = s.q2_bar' .+ s.q2_prime

    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

    # Pre-calculate anomalies and means
    q1_zm = mean(q1, dims=1)
    q2_zm = mean(q2, dims=1)

    q1_anom = q1 .- q1_bg'
    q2_anom = q2 .- q2_bg'

    mid_ind = ceil(length(y)/2) + 0.5
    pm_ind = ceil((3.5 * WC/Ly) * length(y)) + 0.5
    y_lower_lim = y[1]  
    y_upper_lim = y[end] 
    
    plotname = "BCI_snapshots"
    
    fig, ax = plt.subplots(2, 5, figsize=(20, 10), width_ratios=[1., 0.4, 1., 0.4, 0.4])
    fig.tight_layout(pad=4.0)
    fsize=16

    cf1 = ax[5].contourf(x, y, v1', cmap=PyPlot.cm.bwr, levels=[-1.75, -1.0, -0.25, -0.01, 0.01, 0.25, 1.0, 1.75], extend="both")  
    ax[5].set_title(L"v_{1}^{\prime}", fontsize=fsize)

    cbcf1_ax = fig.add_axes([0.395,0.98,0.215,0.03])
    cbar_cf1 = fig.colorbar(cf1, ax=ax[5], location="top", pad=0.2, cax = cbcf1_ax)
    cbar_cf1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    cbar_cf1.ax.tick_params(labelsize=fsize, rotation=45)

    ax[6].contourf(x, y, v2', cmap=PyPlot.cm.bwr, levels=[-1.75, -1.0, -0.25, -0.01, 0.01, 0.25, 1.0, 1.75])  
    ax[6].set_title(L"v_{2}^{\prime}", fontsize=fsize)

    BCI_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "black", "green", "black", "red", "black", "yellow", "black", "blue"])    

    q1_bg_loc = q1_bg .+ beta * y'
    q2_bg_loc = q2_bg .+ beta * y'

    norm1=plt.Normalize(minimum(q1_bg_loc), maximum(q1_bg_loc))
    levs1=collect(range(-maximum(abs.(q1_anom)), maximum(abs.(q1_anom)), 10))
    levs1 = [levs1[1:4]; levs1[7:end]]
    
    pc1=ax[1].pcolormesh(x, y, (q1 .+ beta * y')', cmap=BCI_cmap, norm=norm1)
    ax[1].contour(x, y, q1_anom', colors="#65fe08", levels= levs1) 
    ax[1].set_title(L"q_{1} \quad (\mathrm{color:} \, q^\mathrm{total}_{1}, \ \mathrm{lines:} \, q_{1}^{\prime})", fontsize=fsize)

    cbpc1_ax = fig.add_axes([0.075,0.98,0.215,0.03])
    cbar_pc1 = fig.colorbar(pc1, ax=ax[1], location="top", pad=0.2, cax=cbpc1_ax)

    norm2=plt.Normalize(minimum(q2_bg_loc), maximum(q2_bg_loc))
    levs2=collect(range(-maximum(abs.(q2_anom)), maximum(abs.(q2_anom)), 10))
    levs2 = [levs2[1:4]; levs2[7:end]]
    
    pc2=ax[2].pcolormesh(x, y, (q2 .+ beta * y')', cmap=BCI_cmap, norm=norm2)
    ax[2].contour(x, y, q2_anom', colors="#65fe08", levels= levs2) 
    ax[2].set_title(L"q_{2} (\mathrm{color:} \, q^\mathrm{total}_{2}, \ \mathrm{lines:} \, q_{2}^{\prime})", fontsize=fsize)

    cbpc2_ax = fig.add_axes([0.075,-0.05,0.215,0.03])
    cbar_pc2 = fig.colorbar(pc2, ax=ax[2], location="top", pad=0.2, cax=cbpc2_ax)

    for axn in [ax[1], ax[2], ax[5], ax[6]]
        axn.set_xlabel("x", fontsize=fsize)
        axn.set_ylabel("y", fontsize=fsize)
        axn.set_ylim(y_lower_lim, y_upper_lim)
    end

    ax[3].plot(d_dy(mean(q1 .+ beta * y', dims=1), dy)', y, "k-", label=L"\mathrm{inst.}")
    ax[4].plot(d_dy(mean(q2 .+ beta * y', dims=1), dy)', y, "k-", label=L"\mathrm{inst.}")

    ax[3].plot(d_dy(mean(q1_bg_loc, dims=1), dy)', y, "r--", label=L"\mathrm{bckgrd}")
    ax[4].plot(d_dy(mean(q2_bg_loc, dims=1), dy)', y, "r--", label=L"\mathrm{bckgrd.}")

    ax[3].set_title(L"\partial_y \overline{q}_{1}", fontsize=fsize)
    ax[4].set_title(L"\partial_y \overline{q}_{1}", fontsize=fsize)

    for axn in ax[3:4]
        axn.set_xlabel("x", fontsize=fsize)
        axn.set_ylim(y_lower_lim, y_upper_lim)
        axn.legend(loc="upper right", fontsize=fsize, bbox_to_anchor=[1.35, 1.025])
    end

    ax[7].plot(U_bg', y, "r--", label=L"\mathrm{bckgrd.}")
    ax[7].plot(mean(u1, dims=1)', y, "k-", label=L"\overline{u}_{1}")
    ax[7].set_title(L"\mathrm{Layer \ 1 \ zonal \ flow}", fontsize=fsize)
    ax[7].set_xlim(-0.5, 2.5)

    ax[8].plot(zeros(size(U_bg)), y, "r--", label=L"\mathrm{bckgrd.}")
    ax[8].plot(mean(u2, dims=1)', y, "k-", label=L"\overline{u}_{2}")
    ax[8].set_title(L"\mathrm{Layer \ 2 \ zonal \ flow}", fontsize=fsize)
    ax[8].set_xlim(-0.5, 2.5)

    for axn in ax[7:8]
        axn.set_xlabel("x", fontsize=fsize)
        axn.set_ylim(y_lower_lim, y_upper_lim)
        axn.legend(loc="upper right", fontsize=fsize, bbox_to_anchor=[1.35, 1.025])
    end

    # Track Kinetic Energy
    push!(KE1_hist, mean((u1 .- mean(u1, dims=1)).^2 .+ (v1 .- mean(v1, dims=1)).^2))
    push!(KE2_hist, mean((u2 .- mean(u2, dims=1)).^2 .+ (v2 .- mean(v2, dims=1)).^2))

    n_current = length(KE1_hist)
    th = collect(range(dt, n_current*dt*ell, length=n_current)) # Approximate time axis based on push! count
    
    ax[9].plot(th, KE1_hist, "k-")
    ax[9].set_title(L"\langle \mathrm{KE}_{1} \rangle", fontsize=fsize)
    ax[9].set_yscale("log")
    plt.grid()

    ax[10].plot(th, KE2_hist, "k-")
    ax[10].set_title(L"\langle \mathrm{KE}_{2} \rangle", fontsize=fsize)
    ax[10].set_yscale("log")
    plt.grid()

    for axn in [ax[9], ax[10]]
        axn.set_xlim(dt, nt*dt)
        axn.set_xlabel("Time [nondim.]", fontsize=fsize)
        axn.set_ylim(1e-7, 0.5)
    end

    savename = @sprintf("%s_%04d.png", joinpath(fig_path, plotname), ell)
    PyPlot.savefig(savename, bbox_inches="tight")
    PyPlot.close()
end





function plot_time_averaged_wake(out_cfg::OutputConfig, t_spinup::Float64)
    println("Calculating time-mean streamfunction...")
    
    # 1. Find all saved .jld2 data files
    data_dir = out_cfg.save_path
    files = filter(f -> endswith(f, ".jld2"), readdir(data_dir, join=true))
    
    ψ1_sum = nothing
    ψ2_sum = nothing
    count = 0
    
    # 2. Loop through and accumulate fields past the spin-up time
    for file in files
        raw_data = load(file)
        
        # Unpack the nested dictionary if JLD2 saved it under the variable name "jld_data"
        data = haskey(raw_data, "jld_data") ? raw_data["jld_data"] : raw_data
        
        # Skip any arbitrary .jld2 files in the folder that don't have our expected keys
        if !haskey(data, "t") || !haskey(data, "ψ1") || !haskey(data, "ψ2")
            continue
        end
        
        t = data["t"]
        
        if t >= t_spinup
            if ψ1_sum === nothing
                # Initialize arrays on the first valid file
                ψ1_sum = zeros(size(data["ψ1"]))
                ψ2_sum = zeros(size(data["ψ2"]))
            end
            
            ψ1_sum .+= data["ψ1"]
            ψ2_sum .+= data["ψ2"]
            count += 1
        end
    end
    
    if count == 0
        println("No valid files found past t_spinup = $t_spinup. Try a lower spin-up time!")
        return
    end
    
    # 3. Calculate the time mean
    ψ1_mean = ψ1_sum ./ count
    ψ2_mean = ψ2_sum ./ count
    
    # 4. Subtract the zonal mean to isolate the stationary waves
    ψ1_eddy = ψ1_mean .- mean(ψ1_mean, dims=1)
    ψ2_eddy = ψ2_mean .- mean(ψ2_mean, dims=1)
    
    # 5. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1lim = maximum(abs.(ψ1_eddy))
    pc1 = ax[1].contourf(ψ1_eddy', cmap=PyPlot.cm.PiYG, vmin=-ax1lim, vmax=ax1lim, levels=15)
    ax[1].set_title("Time-Mean ψ1 Anomaly (Upper Layer)")
    plt.colorbar(pc1, ax=ax[1])
    
    ax2lim = maximum(abs.(ψ2_eddy))
    pc2 = ax[2].contourf(ψ2_eddy', cmap=PyPlot.cm.PiYG, vmin=-ax2lim, vmax=ax2lim, levels=15)
    ax[2].set_title("Time-Mean ψ2 Anomaly (Lower Layer)")
    plt.colorbar(pc2, ax=ax[2])
    
    for axn in ax
        axn.set_xlabel("x (grid points)")
        axn.set_ylabel("y (grid points)")
    end
    
    # Save the figure
    savename = joinpath(out_cfg.fig_path, "Time_Averaged_Wake.png")
    PyPlot.savefig(savename, bbox_inches="tight")
    PyPlot.close()
    
    println("Time-averaged wake plotted using $count snapshots! Saved to: $savename")
end



#############################################################################
############################################################################
function save_all(suite::DiagnosticSuite, filename::String)
    println("\n========== Commencing Global Data Dump ==========")
    
    # Create a master dictionary to hold all diagnostic groups
    dump_dict = Dict{String, Any}()
    
    # Loop through every registered diagnostic in the suite
    for (diag_name, diag) in suite.diags
        # Create a sub-dictionary for this specific diagnostic
        diag_group = Dict{String, Any}()
        
        # Dynamically grab every field (times, CBC, v1ζ1, etc.) and its data
        for field in fieldnames(typeof(diag))
            diag_group[string(field)] = getfield(diag, field)
        end
        
        # Store it under the diagnostic's name (e.g., "Pseudomomentum")
        dump_dict[string(diag_name)] = diag_group
    end
    
    # Save the master dictionary to the JLD2 file
    jldsave(filename; jld_data=dump_dict)
    
    println("Saved all diagnostics to $filename")
    println("==================== Data Dump Complete ====================\n")
end


