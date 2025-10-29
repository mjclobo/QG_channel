################################################################################
# Initialize
################################################################################

struct ModelParams
    Nx::Int
    Ny::Int
    nt::Int
    Lx::Float64
    Ly::Float64
    dt::Float64
    beta::Float64
    f0::Float64
    g::Float64
    H::Vector{Float64}
    ρ0::Float64
    Δρ::Float64
    ν::Float64
    r::Float64
    α::Float64
    U0::Float64
    WC::Float64
end



function half_Hann_window(n, L; rev=false)
    if rev==false
        return @. 0.5 * (1 - cos(2π * n / L / 2))
    else
        hHw = @. 0.5 * (1 - cos(2π * n / L / 2))
        return reverse(hHw)
    end
end

function cumtrapz(X::T, Y::T) where {T <: AbstractVector}
    # Check matching vector length
    @assert length(X) == length(Y) "Input vectors X and Y must have the same length."

    # Initialize Output
    out = similar(X)
    out[1] = 0.0 # Start with a cumulative integral of 0 at the first point

    # Iterate over arrays to calculate cumulative integral
    for i in 2:length(X)
        out[i] = out[i-1] + 0.5 * (X[i] - X[i-1]) * (Y[i] + Y[i-1])
    end

    # Return output
    return out
end

function Lee1997_bg_jet(U0, WC; σ=4)
    dy = y[2] - y[1]

    WS = (1 - 2*WC/Ly)/2    # width of a ``side'', i.e., the distance in the y direction over which background flow decays from U0 to zero; normalized from 0 to 1
    if WS>0.5
        error("WC must be greater than zero")
    end

    # First build U profile
    U = zeros(length(y))
    y_cent = y .+ dy/2

    upper_jet_bound = ceil(Int, WS * length(y))
    lower_jet_bound = ceil(Int, (1 - WS) * length(y))

    # North of jet
    U[1:upper_jet_bound] .= @. U0 * exp(-((y_cent[1:upper_jet_bound] - y_cent[upper_jet_bound])^2) / σ^2) * half_Hann_window(y[1:upper_jet_bound+floor(Int, 2*WC/Ly)], WS * Ly)

    # Middle (flat jet)
    U[upper_jet_bound+1:lower_jet_bound-1] .= U0

    # South of jet
    U[lower_jet_bound:end] .= reverse(U[1:upper_jet_bound])

    # Numerically integrate to get ψ(y)
    ψ_bg = -cumtrapz(y, U)  # U = -dψ/dy ⇒ ψ = -∫ U dy

    return ψ_bg, U
end

# function Lee1997_bg_jet(U0, y, Ly, WC)
#     dy = y[2] - y[1]

#     # First build U profile
#     U = zeros(length(y))
#     y_cent = y .+ dy/2
#     s = 0.25 * Ly  # from Lee (1997)

#     upper_jet_bound = ceil(Int, WC * length(y))
#     lower_jet_bound = ceil(Int, (1 - WC) * length(y))

#     # North of jet
#     U[1:upper_jet_bound] .= @. U0 * exp(-((y_cent[1:upper_jet_bound] - WC * Ly)^2) / s^2) * half_Hann_window(y[1:upper_jet_bound], WC * Ly)

#     # Middle (flat jet)
#     U[upper_jet_bound+1:lower_jet_bound-1] .= U0

#     # South of jet
#     U[lower_jet_bound:end] .= @. U0 * exp(-((y_cent[lower_jet_bound:end] - (1 - WC) * Ly)^2) / s^2)
#     U[lower_jet_bound:end] .= U[lower_jet_bound:end] .* half_Hann_window(y[1:upper_jet_bound], WC * Ly; rev=true)

#     # Numerically integrate to get ψ(y)
#     ψ_bg = -cumtrapz(y, U)  # U = -dψ/dy ⇒ ψ = -∫ U dy

#     return ψ_bg', U
# end



################################################################################
# Functions for restart and building psi of time
################################################################################


function extract_time(filename; file_strings=false)
    # Find the starting index of the search string
    idx_start = findfirst("_t", filename)[end] + 1
    idx_end   = findfirst(".jld", filename)[1] - 1

    time = filename[idx_start:idx_end]
    if file_strings==false
        return parse(Float64, time)
    else
        return filename[1:idx_start-1], filename[idx_end+1:end]
    end
end

function define_t_of_saved_files(Ny, save_path)
    ψfiles = readdir(save_path)
    t_array = zeros(length(ψfiles))

    for (i, file) in enumerate(ψfiles)
        t_array[i] = extract_time(save_path*file)
    end

    return sort(t_array)
end

function construct_psi_of_t(t_array, Ny_saved, save_path)
    ψ1_of_t = zeros(Nx, Ny_saved, length(t_array))
    ψ2_of_t = zeros(Nx, Ny_saved, length(t_array))
    file_start, file_end = extract_time(readdir(save_path)[1]; file_strings=true)

    for (i, t) in enumerate(t_array)
        savedata = load(save_path * file_start * string(t) * file_end)

        ψ1_of_t[:,:,i] = savedata["jld_data"]["ψ1"]
        ψ2_of_t[:,:,i] = savedata["jld_data"]["ψ2"]
    end
    return ψ1_of_t, ψ2_of_t
end

################################################################################
# Basic time stepping loop
################################################################################


function run_model(q1, q2, t0, params; timestepper="RK4", output_every=500)
    start_time = time()
    # To turn off the PyPlot GUI
    PyPlot.pygui(false)

    # choose time stepping method; have to use anoNymous function bc "lexical scoping" of Julia...
    # if timestepper=="RK4"
    #     println("Using RK4 time stepper")
    #     ts = q1, q2, dt -> rk4(q1, q2, dt)
    # elseif timestepper=="RK4_int"
    #     println("Using RK4 + integrating factor time stepper")
    #     ts = q1, q2, dt -> rk4_integrating_factor(q1, q2, dt)
    # else
    #     error("You're asking for a timestepping method that doesn't exist.")
    # end
    # timestep(q1, q2, dt) = ts(q1, q2, dt)

    # calculating y indices for prescribed meridional width of save domain
    save_ind_start = floor(Int, Ny * y_width / 2)
    save_ind_end   = floor(Int, Ny * (1 - y_width / 2))

    # defining a couple of counters
    cnt=1
    ell=1

    ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)
    
    KE1 = [mean((u1 .- mean(u1, dims=1)).^2 .+ (v1 .- mean(v1, dims=1)).^2)]
    KE2 = [mean((u2 .- mean(u2, dims=1)).^2 .+ (v2 .- mean(v2, dims=1)).^2)]

    for n = 1:nt

        q1, q2 = rk4(q1, q2, dt)

        q1[:, 1] .= q1_bg[:, 1]
        q1[:, end] .= q1_bg[:, end]
        q2[:, 1] .= q2_bg[:, 1]
        q2[:, end] .= q2_bg[:, end]


        if mod(n, output_every) == 0      # output a message
            ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)

            if isnan(ψ1[2,2])
                error("Psi is NaN")
            else
                u1, v1 = u_from_psi(ψ1)
                u2, v2 = u_from_psi(ψ2)

                cfl = dt * maximum([maximum([u1; u2]) / dx, maximum([v1; v2]) / dy])

                elapsed_time = time() - start_time

                # modified from GophysicalFlows.jl ex
                log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE1 avg.: %.4e, KE2 avg.: %.4e, ens1: %.4e, ens2: %.4e, walltime: %.2f min",
                n, t0+n*dt, cfl, mean(u1.^2 .+ v1.^2), mean(u2.^2 .+ v2.^2), sum(L2D(ψ1).^2), sum(L2D(ψ2).^2), elapsed_time/60)

                println(log)

            end
        end

        if mod(n, save_every) == 0          # save streamfunction fields
            if save_bool==true # && n >= start_saving*nt
                ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)

                save_streamfunction(save_path, ψ1[:,save_ind_start:save_ind_end], ψ2[:,save_ind_start:save_ind_end], t0+n*dt, params)
                cnt+=1

            end
        end

        if mod(n, plot_every) == 0          # plot whatever is in save_basic_anim_panel() function
            if plot_basic_bool==true
                ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # (q1, q2)

                save_basic_anim_panel(fig_path, ell, q1, q2, ψ1, ψ2, U_bg)
            end
            if plot_BCI_bool==true
                save_growth_plot(fig_path, ell, q1, q2, U_bg, n, nt, KE1, KE2)
            end
            ell+=1
        end

    end

    if save_last==true
        ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy)  # (q1, q2)

        save_streamfunction(save_path, ψ1, ψ2, t0+nt*dt, params)
    end

    # To turn the PyPlot GUI back on
    PyPlot.pygui(true)
end




function run_model_decomp(q1_bar, q2_bar, q1_prime, q2_prime, t0, params; timestepper="RK4", output_every=500)
    start_time = time()
    # To turn off the PyPlot GUI
    PyPlot.pygui(false)

    # choose time stepping method; have to use anoNymous function bc "lexical scoping" of Julia...
    # if timestepper=="RK4"
    #     println("Using RK4 time stepper")
    #     ts = q1, q2, dt -> rk4(q1, q2, dt)
    # elseif timestepper=="RK4_int"
    #     println("Using RK4 + integrating factor time stepper")
    #     ts = q1, q2, dt -> rk4_integrating_factor(q1, q2, dt)
    # else
    #     error("You're asking for a timestepping method that doesn't exist.")
    # end
    # timestep(q1, q2, dt) = ts(q1, q2, dt)

    # calculating y indices for prescribed meridional width of save domain
    save_ind_start = floor(Int, Ny * y_width / 2)
    save_ind_end   = floor(Int, Ny * (1 - y_width / 2))

    # defining a couple of counters
    cnt=1
    ell=1

    ψ1_bar = invert_qg_pv_bar(PVBS, q1_bar, ψ1_bg[1])
    ψ2_bar = invert_qg_pv_bar(PVBS, q2_bar, ψ2_bg[1])

    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)
    u1_prime, v1_prime = u_from_psi(ψ1_prime)
    u2_prime, v2_prime = u_from_psi(ψ2_prime)
    
    KE1 = [mean((u1_prime .- mean(u1_prime, dims=1)).^2 .+ (v1_prime .- mean(v1_prime, dims=1)).^2)]
    KE2 = [mean((u2_prime .- mean(u2_prime, dims=1)).^2 .+ (v2_prime .- mean(v2_prime, dims=1)).^2)]

    for n = 1:nt

        q1_prime, q2_prime, q1_bar, q2_bar = rk4_coupled(q1_prime, q2_prime, q1_bar, q2_bar, dt)

        if mod(n, output_every) == 0      # output a message
            ψ1_bar = invert_qg_pv_bar(PVBS, q1_bar, ψ1_bg[1])
            ψ2_bar = invert_qg_pv_bar(PVBS, q2_bar, ψ2_bg[1])
        
            ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

            ψ1 = ψ1_bar' .+ ψ1_prime
            ψ2 = ψ2_bar' .+ ψ2_prime
        
            if isnan(ψ1[2,2])
                error("Psi is NaN")
            else
                u1, v1 = u_from_psi(ψ1)
                u2, v2 = u_from_psi(ψ2)

                cfl = dt * maximum([maximum([u1; u2]) / dx, maximum([v1; v2]) / dy])

                elapsed_time = time() - start_time

                # modified from GophysicalFlows.jl ex
                log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, KE1 avg.: %.4e, KE2 avg.: %.4e, ens1: %.4e, ens2: %.4e, walltime: %.2f min",
                n, t0+n*dt, cfl, mean(u1.^2 .+ v1.^2), mean(u2.^2 .+ v2.^2), sum(L2D(ψ1).^2), sum(L2D(ψ2).^2), elapsed_time/60)

                println(log)

            end
        end

        if mod(n, save_every) == 0          # save streamfunction fields
            if save_bool==true # && n >= start_saving*nt
                ψ1_bar = invert_qg_pv_bar(PVBS, q1_bar, ψ1_bg[1])
                ψ2_bar = invert_qg_pv_bar(PVBS, q2_bar, ψ2_bg[1])
            
                ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)
            
                ψ1 = ψ1_bar' .+ ψ1_prime
                ψ2 = ψ2_bar' .+ ψ2_prime

                save_streamfunction(save_path, ψ1[:,save_ind_start:save_ind_end], ψ2[:,save_ind_start:save_ind_end], t0+n*dt, params)
                cnt+=1

            end
        end

        if mod(n, plot_every) == 0          # plot whatever is in save_basic_anim_panel() function
            if plot_basic_bool==true
                ψ1_bar = invert_qg_pv_bar(PVBS, q1_bar, ψ1_bg[1])
                ψ2_bar = invert_qg_pv_bar(PVBS, q2_bar, ψ2_bg[1])
            
                ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)
            
                save_basic_anim_panel(fig_path, ell, q1_bar' .+ q1_prime, q2_bar' .+ q2_prime, ψ1_bar' .+ ψ1_prime, ψ2_bar' .+ ψ2_prime, U_bg)
            end
            if plot_BCI_bool==true
                save_growth_plot(fig_path, ell, q1_bar' .+ q1_prime, q2_bar' .+ q2_prime, U_bg, n, nt, KE1, KE2)
            end
            ell+=1
        end

    end

    if save_last==true
        ψ1_bar = invert_qg_pv_bar(PVBS, q1_bar, ψ1_bg[1])
        ψ2_bar = invert_qg_pv_bar(PVBS, q2_bar, ψ2_bg[1])
    
        ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)
    
        save_streamfunction(save_path, ψ1_bar' .+ ψ1_prime, ψ2_bar' .+ ψ2_prime, t0+nt*dt, params)
    end

    # To turn the PyPlot GUI back on
    PyPlot.pygui(true)
end

