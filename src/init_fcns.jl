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
    trans::Float64
end


struct BTParams
    Nx::Int
    Ny::Int
    nt::Int
    Lx::Float64
    Ly::Float64
    dt::Float64
    beta::Float64
    Ld::Float64
    ν::Float64
    r::Float64
    U0::Float64
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

function Lee1997_bg_jet(U0, WC; σ=6.0)

    WS = (1 - 2*WC/Ly)/2    # width of a ``side'', i.e., the distance in the y direction over which background flow decays from U0 to zero; normalized from 0 to 1
    if WS>0.5
        error("WC must be greater than zero")
    end

    # First build U profile
    U = zeros(length(y))

    for (i, yi) in enumerate(y)
        if abs(yi) < WC
            U[i] = U0
        else
            U[i] = U0 * exp(-(abs(yi) - WC)^2 * σ^-2)
        end
    end

    upper_jet_bound = findfirst(x->x==U0, U)
    lower_jet_bound = findlast(x->x==U0, U)

    # this is also a good option!
    # U .= U0 .* 0.5 .* (1 .- tanh.(5 .* (abs.(y) .- WC) ./ s))

    # Numerically integrate to get ψ(y)
    ψ_bg = -cumtrapz(y, U)  # U = -dψ/dy ⇒ ψ = -∫ U dy

    if upper_jet_bound==nothing
        upper_jet_bound = 1
        lower_jet_bound = length(y)
    end

    return ψ_bg, U, upper_jet_bound, lower_jet_bound
end


function blended_transport_jet(y; y0=0.0, T=35.0, W=10.0, σ=6.0, trans=0.0, U0=0.0)

    # --- 1. Flat core + smooth Gaussian shoulders ---
    function U_flat(yi, y0)
        r = abs(yi-y0)
        if r <= W
            return 1.0
        else
            ξ = (r - W) / σ
            return exp(-ξ^4)   # quartic Gaussian
        end
    end

    # --- 2. Pavan–Held profile (rescaled to comparable width) ---
    σp = W / 1.0   # tunable but keeps widths comparable
    U_ph(yi, y0) = sech((yi-y0) / σp)^2

    # --- 3. Geometric blend ---
    Ũ = similar(y)
    for (i, yi) in enumerate(y)
        Ũ[i] = U_flat(yi, y0)^(1-trans) * U_ph(yi, y0)^trans
    end

    # --- Hanning taper (only outside |y| > W) ---
    L = maximum(abs.(y))   # half-domain size

    taper = similar(y)

    for (i, yi) in enumerate(y)
        r = abs(yi - y0)
        if r <= W
            taper[i] = 1.0
        else
            ξ = (r - W) / (L - W)   # map to [0,1]
            taper[i] = 0.5 * (1 + cos(pi * ξ))  # Hanning window
        end
    end

    # Apply taper
    Ũ .= Ũ .* taper

    # --- 4. Normalize to fixed transport ---
    if T==0
        U = Ũ
    else
        T̃ = sum(Ũ) * dy
        U = T .* Ũ ./ T̃
    end

    if U0!=0.0
        U = U0 .* (U ./ maximum(U))
    end

    ψ_bg = -cumtrapz(y, U) 

    upper_jet_bound = 1
    lower_jet_bound = length(y)

    return ψ_bg, U, upper_jet_bound, lower_jet_bound
end


function double_jet(y; sep=15, σ=3.0, Wtaper = 35)
    # sep is a separation parameter that measures how far the respective
    # baroclinic zone centers are from the center of the domain
    Ny = length(y)
    mid_ind = Int(Ny/2)
    ## y MUST BE CENTERED ON ZERO

    # --- 1. Flat core + ultra-smooth Gaussian shoulders ---
    function Gaussian_U(yi, cent, σ)
        return @. exp(-(yi - cent)^2 / (2*σ^2))
    end

    # need to align cent with nearest discrete y value
    sep_proj = y[argmin(abs.(y .- sep))]

    # define background flow profile
    # U = zeros(Ny)

    # U[1:mid_ind] = Gaussian_U(y, -sep_proj, σ)[1:mid_ind]
    # U[mid_ind+1:end] = Gaussian_U(y, sep_proj, σ)[mid_ind+1:end]

    U = Gaussian_U(y, -sep_proj, σ) .+ Gaussian_U(y, sep_proj, σ)

    ## tapering to zero at domain edges
    L = maximum(abs.(y))   # half-domain size

    taper = similar(y)

    for (i, yi) in enumerate(y)
        r = abs(yi)
        if r <= Wtaper
            taper[i] = 1.0
        else
            ξ = (r - Wtaper) / (L - Wtaper)   # map to [0,1]
            taper[i] = 0.5 * (1 + cos(pi * ξ))  # Hanning window
        end
    end

    U = U .* taper

    # finding streamfunction from U
    ψ_bg = -cumtrapz(y, U) 

    upper_jet_bound = 1
    lower_jet_bound = Ny

    return ψ_bg, U ./ maximum(U), upper_jet_bound, lower_jet_bound
end

function smooth_core_jet(y; U0=1.0, W=20.0)

    Ny = length(y)
    U = similar(y)

    for (i, yi) in enumerate(y)

        a = abs(yi)

        if a >= W
            U[i] = 0.0
        else
            # smooth top-hat envelope
            envelope = 0.5 * (1 + cospi(a / W))

            # “linear-core shaping”
            core = 1 - (a / W)^2

            # blend them smoothly
            U[i] = U0 * envelope * core
        end
    end
    
    # finding streamfunction from U
    ψ_bg = -cumtrapz(y, U) 

    upper_jet_bound = 1
    lower_jet_bound = Ny

   return ψ_bg, U ./ maximum(U), upper_jet_bound, lower_jet_bound
end

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

function run_model_decomp(model::QGModel, suite::DiagnosticSuite, out_cfg::OutputConfig, 
                          ψ_diff_bg, U_bg, t0, params; 
                          output_every=200, save_ind_start=1, save_ind_end=params.Ny, 
                          t_start_diag=250, topo_PV=0.0, wind_curl=0.0, q1_STJ_target=0.0, q2_STJ_target=0.0, update_bg=false, fix_zonal_mean=false)
    
    start_time = time()
    PyPlot.pygui(false) # Turn off PyPlot GUI

    # Unpack grid variables for safety and clarity
    Nx, Ny = params.Nx, params.Ny
    dx, dy = params.Lx / Nx, params.Ly / Ny
    s = model.state
    o = model.ops
    dt = params.dt 

    # Pre-allocate 2D fields in the outer scope
    # ψ1_bg_2D = zeros(Nx, Ny)
    # q1_bg_2D = zeros(Nx, Ny)

    if fix_zonal_mean == true
        # Extract the 1D background slice directly from the matrix passed into the function
        # This ensures it exactly matches your main execution script
        ψ1_bg_1D = vec(ψ_diff_bg[1, :])

        # 1. Populate the 1D bar profiles for initialization
        s.ψ1_bar .= ψ1_bg_1D
        s.ψ2_bar .= zeros(Ny)

        # Compute the corresponding 1D bar PV using 1D operator
        q1_bg_1D, q2_bg_1D = compute_qg_pv_bar(s.ψ1_bar, s.ψ2_bar; lap_op=o.L1D)
        s.q1_bar .= q1_bg_1D
        s.q2_bar .= q2_bg_1D

        # 2. Build the true 2D matrices required for the rhs_full framework
        ψ1_bg_2D = repeat(ψ1_bg_1D', Nx, 1)
	q1_bg_2D = repeat(q1_bg_1D', Nx, 1)
	q2_bg_2D = repeat(q2_bg_1D', Nx, 1)

	# 1. Compute the second derivative of the velocity profiles using your 1D operator
	# (Assuming L1D acts as d^2/dy^2 on a 1D vector)
	d2U1_dy2 = model.ops.L1D(U_bg)
	d2U2_dy2 = zeros(Ny)  # Since U2_bg is zero

	# 2. Correctly use the background velocities (U) for the stretching term!
	q1_bg_grad_1D = beta .- d2U1_dy2 .+ F1 .* (U_bg .- zeros(Ny))
	q2_bg_grad_1D = beta .- d2U2_dy2 .+ F2 .* (zeros(Ny) .- U_bg)

	# 3. Repeat to 2D matrices for your rhs_full function
	q1_bg_grad_2D = repeat(q1_bg_grad_1D', Nx, 1)
	q2_bg_grad_2D = repeat(q2_bg_grad_1D', Nx, 1)
    else
	# dummy variables for plotting routine...should fix this at smoe point!
        ψ1_bg_2D = nothing
        q1_bg_2D = nothing
        q2_bg_2D = nothing
    end

    cnt = 1
    ell = 1

    # Initialize empty history arrays in case BCI plotting is turned on
    KE1_hist = Float64[]
    KE2_hist = Float64[]

    # Capture initial background PV for the BCI plots
    q1_bg_init = copy(s.q1_bar)
    q2_bg_init = copy(s.q2_bar)

    # Main Time Loop
    for n = 1:params.nt
        t = t0 + n * dt

        # Potentially update background target field
        if update_bg
            ψ1_bg_1D, U_bg, _, _ = ψ_diff_bg_of_t(t)
            
            # Update the matrices dynamically for both frameworks
            ψ1_bg_2D .= repeat(ψ1_bg_1D', Nx, 1)
            q1_bg_1D, q2_bg_1D = compute_qg_pv_bar(ψ1_bg_1D, zeros(Ny); lap_op=o.L1D)
            q1_bg_2D .= repeat(q1_bg_1D', Nx, 1)
	    q2_bg_2D .= repeat(q2_bg_1D', Nx, 1)
            
            ψ_diff_bg = copy(ψ1_bg_1D) 

        end

        # --- Time Step Selection ---
        if fix_zonal_mean == false
            rk4_coupled(model, ψ_diff_bg, topo_PV, wind_curl, q1_STJ_target, q2_STJ_target, dt)
        else
            rk4_fixed_bg(model, ψ1_bg_2D, q1_bg_grad_2D, q2_bg_grad_2D, topo_PV, dt)    # assumes NO WIND FORCING
        end

        # Diagnostics
        if t > t_start_diag
            compute_all!(suite, model, t, n)
        end

        # Logging
        if mod(n, output_every) == 0
            # Ensure streamfunctions are strictly synced with the latest PV
            s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)
            s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)

            # Reconstruct absolute streamfunction for logging diagnostics
            if fix_zonal_mean == false
                ψ1 = s.ψ1_bar' .+ s.ψ1_prime 
                ψ2 = s.ψ2_bar' .+ s.ψ2_prime 
            else
                ψ1 = ψ1_bg_2D .+ s.ψ1_prime
                ψ2 = s.ψ2_prime
            end

            if isnan(ψ1[2,2])
                error("Psi is NaN at step $n")
            else
                u1, v1 = u_from_psi(ψ1)
                u2, v2 = u_from_psi(ψ2)

                total_nrg = 0.5 * sum(u1.^2 .+ v1.^2) + 0.5 * sum(u2.^2 .+ v2.^2) + sum((0.5 * (ψ1 .- ψ2)).^2)
                
                cfl = dt * maximum([maximum([u1; u2]) / dx, maximum([v1; v2]) / dy])
                elapsed_time = time() - start_time

                log = @sprintf("step: %04d, t: %.1f d, cfl: %.2f, KE1 avg.: %.4e, KE2 avg.: %.4e, KE tot.: %.4e, ens1: %.4e, ens2: %.4e, walltime: %.2f min",
                               n, t/3600/24, cfl, mean(u1.^2 .+ v1.^2), mean(u2.^2 .+ v2.^2), total_nrg, sum(o.L2D(ψ1).^2), sum(o.L2D(ψ2).^2), elapsed_time/60)
                println(log)
            end
        end

        # Save Streamfunctions
        if out_cfg.save_bool && mod(n, out_cfg.save_every) == 0
            if fix_zonal_mean == false
                ψ1 = s.ψ1_bar' .+ s.ψ1_prime
                ψ2 = s.ψ2_bar' .+ s.ψ2_prime
            else
                ψ1 = ψ1_bg_2D .+ s.ψ1_prime
                ψ2 = s.ψ2_prime
            end
            save_streamfunction(out_cfg.save_path, ψ1[:, save_ind_start:save_ind_end], ψ2[:, save_ind_start:save_ind_end], t, params)
            cnt += 1
        end

        # Plotting
        if mod(n, out_cfg.plot_every) == 0
	    if out_cfg.plot_basic_bool
        	save_basic_anim_panel(out_cfg.fig_path, ell, model, U_bg, t, ψ1_bg_2D, q1_bg_2D, q2_bg_2D, fix_zonal_mean)
    	    end
            if out_cfg.plot_BCI_bool
                save_growth_plot(out_cfg.fig_path, ell, model, U_bg, dt, params.nt, KE1_hist, KE2_hist, q1_bg_init, q2_bg_init, ψ_diff_bg)
            end
            ell += 1
        end
    end

    # Save Final State
    if out_cfg.save_last
        s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)
        s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)
        
        if fix_zonal_mean == false
            ψ1 = s.ψ1_bar' .+ s.ψ1_prime
            ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        else
            ψ1 = ψ1_bg_2D .+ s.ψ1_prime
            ψ2 = s.ψ2_prime
        end
        save_streamfunction(out_cfg.save_path, ψ1, ψ2, t0 + params.nt * dt, params)
    end

    # Perform Global Diagnostic Dump
    if out_cfg.diag_bool
        t_diag = t0 + params.nt * dt
        file_name = struct_to_string(params) * "_diags_t$t_diag.jld2"
        save_all(suite, joinpath(out_cfg.diag_dir, file_name))
    end
end







