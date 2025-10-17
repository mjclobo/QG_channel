
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


function save_basic_anim_panel(fig_path, ell, q1, q2, U_bg)
    plotname = "snapshots"
    ψ1, ψ2 = invert_qg_pv(q1, q2)

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
