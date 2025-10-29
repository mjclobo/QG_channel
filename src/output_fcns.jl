
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
        else
            push!(field_strings, "$field_name"*string(round(field_value, digits=3)))
        end
    end
    return join(field_strings, "_")
end

function save_streamfunction(dir, ψ1, ψ2, t, params)

    file_name = struct_to_string(params) * "_t$t.jld"
    # Save variables to JLD file
    jld_data = Dict("ψ1" => Array(ψ1), "ψ2" => Array(ψ2), "t" => t)
    jldsave(dir * file_name; jld_data)

    println("Saved streamfunction to $file_name")

    # if isfile(dir*"streamfunction_step$(step-30).jld")
    #     rm(dir*"streamfunction_step$(step-30).jld")
    #     println("Deleted: " * dir * "streamfunction_step$(step-30).jld")
    # end

end


function save_basic_anim_panel(fig_path, ell, q1, q2, ψ1, ψ2, U_bg)
    plotname = "snapshots"
    # ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)

    fig, ax = plt.subplots(2, 3, figsize=(14, 10), width_ratios=[1., 1., 0.4])

    ax2lim = maximum(abs.(ψ1 .- mean(ψ1, dims=1)))
    ax[3].contourf(x, y, (ψ1 .- mean(ψ1, dims=1))', cmap=PyPlot.cm.PiYG, vmin=-ax2lim, vmax=ax2lim, levels=7)
    ax[3].set_title("ψ1")

    ax4lim = maximum(abs.(ψ2 .- mean(ψ2, dims=1)))
    ax[4].contourf(x, y, (ψ2 .- mean(ψ2, dims=1))', cmap=PyPlot.cm.PiYG, vmin=-ax4lim, vmax=ax4lim, levels=7)
    ax[4].set_title("ψ2")

    ax1lim = maximum(abs.(q1 .- mean(q1, dims=1)))
    ax[1].pcolormesh(x, y, (q1 .- mean(q1, dims=1))', cmap=PyPlot.cm.bwr, vmin=-ax1lim, vmax=ax1lim)
    ax[1].set_title("q1")

    ax3lim = maximum(abs.(q2 .- mean(q2, dims=1)))
    ax[2].pcolormesh(x, y, (q2 .- mean(q2, dims=1))', cmap=PyPlot.cm.bwr, vmin=-ax3lim, vmax=ax3lim)
    ax[2].set_title("q2")

    # ψ1_init = deepcopy(ψ_diff_bg) # zeros(Nx, Ny) # 
    # ψ2_init = zeros(Nx, Ny)

    # q1_init, q2_init = compute_qg_pv(ψ1_init, ψ2_init)

    # q1_init .+= 1e0 * randn(Nx, Ny)
    # q2_init .+= 1e0 * randn(Nx, Ny)

    # ax1lim = maximum(abs.(q1_init))
    # ax[1].pcolormesh(x, y, (q1)', cmap=PyPlot.cm.bwr, vmin=-ax1lim, vmax=ax1lim)
    # ax[1].set_title("q1")

    # ax3lim = maximum(abs.(q2_init))
    # ax[2].pcolormesh(x, y, (q2)', cmap=PyPlot.cm.bwr, vmin=-ax3lim, vmax=ax3lim)
    # ax[2].set_title("q2")

    for axn in ax[1:4]
        axn.set_xlabel("x")
        axn.set_ylabel("y")
    end

    ###
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

    ax[5].plot(U_bg', y, "r--")
    ax[5].plot(mean(u1, dims=1)', y, "k-")
    ax[5].set_title("U_1 (zonal avg.)")
    ax[5].set_xlim(-1, 5)

    ax[6].plot(zeros(size(U_bg)), y, "r--")
    ax[6].plot(mean(u2, dims=1)', y, "k-")
    ax[6].set_title("U_2 (zonal avg.)")
    ax[6].set_xlim(-1, 5)

    for axn in ax[5:6]
        axn.set_xlabel("x")
        axn.set_ylim(y[1], y[end])
    end

    ###
    local savename = @sprintf("%s_%04d.png", joinpath(fig_path, plotname), ell)
    PyPlot.savefig(savename)

    PyPlot.close()
end



function save_growth_plot(fig_path, ell, q1, q2, U_bg, n, nt, KE1, KE2)
    # defining initial conditions to define colorbar scale
    ψ1_init = deepcopy(ψ_diff_bg)
    ψ2_init = zeros(Nx, Ny)

    q1_init, q2_init = compute_qg_pv(ψ1_init, ψ2_init)

    q1_init .+= 1e0 * randn(Nx, Ny)
    q2_init .+= 1e0 * randn(Nx, Ny)

    # doesn't include noise, but noise should be small anyways, hopefully
    u1_init, v1_init = u_from_psi(ψ1_init)
    u2_init, v2_init = u_from_psi(ψ2_init)

    ########################################################
    mid_ind = ceil(length(y)/2) + 0.5
    pm_ind = ceil((3.5 * WC/Ly) * length(y)) + 0.5
    y_lower_lim = y[1]  # y[Int(mid_ind-pm_ind)]
    y_upper_lim = y[end] # y[Int(mid_ind+pm_ind)]
    ######################################################
    plotname = "BCI_snapshots"
    ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

    if ell==150
        println(maximum(abs.(v1)))
        println(maximum(abs.(v2)))
        println(maximum(abs.(q1 .- mean(q1, dims=1))))
        println(maximum(abs.(q2 .- mean(q2, dims=1))))
    end
    
    #################
    fig, ax = plt.subplots(2, 5, figsize=(20, 10), width_ratios=[1., 0.4, 1., 0.4, 0.4])
    fig.tight_layout(pad=4.0)

    fsize=16

    ax2lim = 3.0 # maximum(abs.(v1_init))
    cf1 = ax[5].contourf(x, y, v1', cmap=PyPlot.cm.bwr, levels=[-1.75, -1.0, -0.25, -0.01, 0.01, 0.25, 1.0, 1.75], extend="both")  # levels=[-2.1, -1.6, -1.1, -0.6, -0.1, 0.1, 0.6, 1.1, 1.6, 2.1])  # collect(range(-2.25, 2.25, 10)))  # levels=[-1.5, -0.5, -0.1, 0.1, 0.5, 1.5])
    ax[5].set_title(L"v_{1}^{\prime}", fontsize=fsize)

    cbcf1_ax = fig.add_axes([0.395,0.98,0.215,0.03])
    cbar_cf1 = fig.colorbar(cf1, ax=ax[5], location="top", pad=0.2, cax = cbcf1_ax)
    cbar_cf1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    cbar_cf1.ax.tick_params(labelsize=fsize, rotation=45)

    ax4lim = 1.1 # maximum(abs.(v2_init))
    ax[6].contourf(x, y, v2', cmap=PyPlot.cm.bwr, levels=[-1.75, -1.0, -0.25, -0.01, 0.01, 0.25, 1.0, 1.75])  # levels=[-2.1, -1.6, -1.1, -0.6, -0.1, 0.1, 0.6, 1.1, 1.6, 2.1], extend="both")  # collect(range(-2.25, 2.25, 10)))  # levels=[-1.5, -0.5, -0.1, 0.1, 0.5, 1.25])
    ax[6].set_title(L"v_{2}^{\prime}", fontsize=fsize)

    ###############################################################################
    BCI_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "black", "green", "black", "red", "black", "yellow", "black", "blue"])    


    # q1_bg_loc = mean(q1, dims=1) .+ beta * y'
    # q2_bg_loc = mean(q2, dims=1) .+ beta * y'

    q1_bg_loc = q1_bg .+ beta * y'
    q2_bg_loc = q2_bg .+ beta * y'
    q1_anom = q1 .- q1_bg
    q2_anom = q2 .- q2_bg

    norm1=plt.Normalize(minimum(q1_bg_loc), maximum(q1_bg_loc))
    levs1=collect(range(-maximum(abs.(q1_anom)), maximum(abs.(q1_anom)), 10))
    levs1 = [levs1[1:4]; levs1[7:end]]
    pc1=ax[1].pcolormesh(x, y, (q1 .+ beta * y')', cmap=BCI_cmap, norm=norm1)
    ax[1].contour(x, y, q1_anom', colors="#65fe08", levels= levs1) # 4.0 .* [-0.75, -0.25, 0.25, 0.75], linewidth=0.5)
    ax[1].set_title(L"q_{1} \quad (\mathrm{color:} \, q^\mathrm{total}_{1}, \ \mathrm{lines:} \, q_{1}^{\prime})", fontsize=fsize)

    cbpc1_ax = fig.add_axes([0.075,0.98,0.215,0.03])
    cbar_pc1 = fig.colorbar(pc1, ax=ax[1], location="top", pad=0.2, cax=cbpc1_ax)
    # cbar_pc1 = fig.colorbar(pc1, ax=ax[1], location="top", pad=0.2, ticks=[minimum(q1_bg_loc), maximum(q1_bg_loc)], cax=cbpc1_ax)
    # cbar_pc1.ax.set_xticklabels([L"\mathrm{min.} \{ \overline{q}_{k} \} ", L"\mathrm{max.} \{ \overline{q}_{k} \} "], fontsize=fsize)

    norm2=plt.Normalize(minimum(q2_bg_loc), maximum(q2_bg_loc))
    levs2=collect(range(-maximum(abs.(q2_anom)), maximum(abs.(q2_anom)), 10))
    levs2 = [levs2[1:4]; levs2[7:end]]
    pc2=ax[2].pcolormesh(x, y, (q2 .+ beta * y')', cmap=BCI_cmap, norm=norm2)
    ax[2].contour(x, y, q2_anom', colors="#65fe08", levels= levs2) # 1.0 .* [-0.75, -0.25, 0.25, 0.75], linewidth=0.5)
    ax[2].set_title(L"q_{2} (\mathrm{color:} \, q^\mathrm{total}_{2}, \ \mathrm{lines:} \, q_{2}^{\prime})", fontsize=fsize)

    cbpc2_ax = fig.add_axes([0.075,-0.05,0.215,0.03])
    cbar_pc2 = fig.colorbar(pc2, ax=ax[2], location="top", pad=0.2, cax=cbpc2_ax)

    for axn in [ax[1], ax[2], ax[5], ax[6]]
        axn.set_xlabel("x", fontsize=fsize)
        axn.set_ylabel("y", fontsize=fsize)
        axn.set_ylim(y_lower_lim, y_upper_lim)
    end

    # ax[2].text(0.5, -0.3,L"\mathrm{color:} \, q^\mathrm{total}_{k}, \ \mathrm{lines:} \, q_{k}^{\prime}", ha="center", va="center", transform=ax[2].transAxes,fontsize=16.)

    ###


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

    ###
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

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


    #############################################################
    push!(KE1, mean((u1 .- mean(u1, dims=1)).^2 .+ (v1 .- mean(v1, dims=1)).^2))
    push!(KE2, mean((u2 .- mean(u2, dims=1)).^2 .+ (v2 .- mean(v2, dims=1)).^2))

    th = collect(range(dt, n*dt, ell+1))
    ax[9].plot(th, KE1, "k-")
    ax[9].set_title(L"\langle \mathrm{KE}_{1} \rangle", fontsize=fsize)
    ax[9].set_yscale("log")
    plt.grid()

    ax[10].plot(th, KE2, "k-")
    ax[10].set_title(L"\langle \mathrm{KE}_{2} \rangle", fontsize=fsize)
    ax[10].set_yscale("log")
    plt.grid()

    for axn in [ax[9], ax[10]]
        axn.set_xlim(dt, nt*dt)
        axn.set_xlabel("Time [nondim.]", fontsize=fsize)
        axn.set_ylim(1e-7, 0.5)
    end

    ###
    local savename = @sprintf("%s_%04d.png", joinpath(fig_path, plotname), ell)
    PyPlot.savefig(savename, bbox_inches="tight") # , transparent=true)

    PyPlot.close()
end

