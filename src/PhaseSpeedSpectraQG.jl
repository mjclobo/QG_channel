# module PhaseSpeedSpectraQG

using FFTW
using LinearAlgebra
using Statistics
using LoopVectorization
using Base.Threads

# export SpectParams, compute_phase_speed_spectra, compute_flux_spectra_block

struct SpectParams
    Nx::Int
    Ny::Int
    nt::Int
    dx::Float64
    dy::Float64
    dt::Float64
    k_vec::Vector{Float64}
    ω_vec::Vector{Float64}
    c_bin_edges::Vector{Float64}
    c_bin_centers::Vector{Float64}
    ncbins::Int
end

function make_spectral_params(params, dx, dy, cmax, ncbins)
    # we pass in dx and dy because dy is NOT Ly/Ny here (because Ny is the number
    # of y-points only in the meridional band where we've output streamfunction data)

    dk = 2π / (params.Nx * dx)
    # Build signed k vector
    ktemp = dk .* (0:(params.Nx-1))
    k_vec = [ m ≤ params.Nx ÷ 2 ? ktemp[m+1] : (ktemp[m+1] - 2π/(dx)) for m in 0:(params.Nx-1) ]

    dω = 2π / (params.nt * params.dt)
    ωtemp = dω .* (0:(params.nt-1))
    ω_vec = [ m ≤ params.nt ÷ 2 ? ωtemp[m+1] : (ωtemp[m+1] - 2π/(params.dt)) for m in 0:(params.nt-1) ]

    c_bin_edges = range(-cmax, stop=cmax, length=ncbins+1) |> collect
    c_bin_centers = (c_bin_edges[1:end-1] .+ c_bin_edges[2:end]) ./ 2

    return SpectParams(params.Nx, params.Ny, params.nt, dx, dy, params.dt, k_vec, ω_vec, c_bin_edges, c_bin_centers, ncbins)
end

function make_block_spectral_params(nt, params, dx, dy, cmax, ncbins)
    # we pass in dx and dy because dy is NOT Ly/Ny here (because Ny is the number
    # of y-points only in the meridional band where we've output streamfunction data)

    dk = 2π / (params.Nx * dx)
    # Build signed k vector
    ktemp = dk .* (0:(params.Nx-1))
    k_vec = [ m ≤ params.Nx ÷ 2 ? ktemp[m+1] : (ktemp[m+1] - 2π/(dx)) for m in 0:(params.Nx-1) ]

    dω = 2π / (nt * params.dt)
    ωtemp = dω .* (0:(nt-1))
    ω_vec = [ m ≤ nt ÷ 2 ? ωtemp[m+1] : (ωtemp[m+1] - 2π/(params.dt)) for m in 0:(nt-1) ]

    c_bin_edges = range(-cmax, stop=cmax, length=ncbins+1) |> collect
    c_bin_centers = (c_bin_edges[1:end-1] .+ c_bin_edges[2:end]) ./ 2

    return SpectParams(params.Nx, params.Ny, nt, dx, dy, params.dt, k_vec, ω_vec, c_bin_edges, c_bin_centers, ncbins)
end


function make_spectral_params_block(Nx, Ny, nt, dx, dy, dt, cmax, ncbins)
    # we pass in dx and dy because dy is NOT Ly/Ny here (because Ny is the number
    # of y-points only in the meridional band where we've output streamfunction data)

    dk = 2π / (params.Nx * dx)
    # Build signed k vector
    ktemp = dk .* (0:(params.Nx-1))
    k_vec = [ m ≤ params.Nx ÷ 2 ? ktemp[m+1] : (ktemp[m+1] - 2π/(dx)) for m in 0:(params.Nx-1) ]

    dω = 2π / (params.nt * dt)
    ωtemp = dω .* (0:(params.nt-1))
    ω_vec = [ m ≤ params.nt ÷ 2 ? ωtemp[m+1] : (ωtemp[m+1] - 2π/(dt)) for m in 0:(params.nt-1) ]

    c_bin_edges = range(-cmax, stop=cmax, length=ncbins+1) |> collect
    c_bin_centers = (c_bin_edges[1:end-1] .+ c_bin_edges[2:end]) ./ 2

    return SpectParams(Nx, Ny, params.nt, dx, dy, dt, k_vec, ω_vec, c_bin_edges, c_bin_centers, ncbins)
end

# derivative in y (and second derivative) optimized
function deriv_y_and_yy!(∂yψ::Array{Float64,3}, ∂yyψ::Array{Float64,3},
                         ψ::Array{Float64,3}, dy::Float64)
    Nx, Ny, nt = size(ψ)
    @assert size(∂yψ) == (Nx, Ny, nt)
    @assert size(∂yyψ) == (Nx, Ny, nt)

    inv2dy = 1.0 / (2*dy)
    invdy2 = 1.0 / (dy^2)

    @threads for t in 1:nt
        # for each time slice, do j loop
        ∂yψ[:,:, t] = d_dy(ψ[:,:,t], dy)
        ∂yyψ[:,:, t] = d_dy(∂yψ[:,:, t], dy)
        # @inbounds @turbo for j in 2:(Ny-1), i in 1:Nx
        #     ∂yψ[i, j, t] = (ψ[i, j+1, t] - ψ[i, j-1, t]) * inv2dy
        #     ∂yyψ[i, j, t] = (ψ[i, j+1, t] - 2.0 * ψ[i, j, t] + ψ[i, j-1, t]) * invdy2
        # end
        # # handle boundaries in y
        # @inbounds for i in 1:Nx
        #     # Neumann/no normal flow => ∂yψ = 0
        #     ∂yψ[i, 1, t] = 0.0
        #     ∂yψ[i, Ny, t] = 0.0
        #     # second derivative one-sided
        #     ∂yyψ[i, 1, t] = (ψ[i, 2, t] - 2.0 * ψ[i, 1, t] + ψ[i, 1, t]) * invdy2
        #     ∂yyψ[i, Ny, t] = (ψ[i, Ny, t] - 2.0 * ψ[i, Ny, t] + ψ[i, Ny-1, t]) * invdy2
        # end
    end

    return nothing
end


"""
Compute phase‑speed spectra (optimized) for eddy heat flux and upper‑layer vorticity flux.

Inputs:
  psi1, psi2 :: Float64 arrays of shape (Nx, Ny, nt)
  params :: SpectParams

Returns:
  F_heat, F_vort :: Float64 arrays (ncbins, Ny)
"""
function compute_phase_speed_spectra(psi1::AbstractArray{Float64,3}, psi2::AbstractArray{Float64,3}, params::SpectParams)

    Nx, Ny, nt = size(psi1)

    ncbins = params.ncbins
    F_heat = zeros(Float64, ncbins, Ny)
    F_vort = zeros(Float64, ncbins, Ny)

    # compute means (zonal + time mean)
    ψ1ztm = mean(mean(psi1, dims=1), dims=3)
    ψ2ztm = mean(mean(psi2, dims=1), dims=3)

    # compute mean fields
    ∂yψ1m = similar(ψ1ztm)
    ∂yyψ1m = similar(ψ1ztm)
    ∂yψ2m = similar(ψ2ztm)
    ∂yyψ2m = similar(ψ2ztm)

    deriv_y_and_yy!(∂yψ1m, ∂yyψ1m, ψ1ztm, params.dy)
    deriv_y_and_yy!(∂yψ2m, ∂yyψ2m, ψ2ztm, params.dy)

    # allocate derivative arrays
    ∂ypsi1  = similar(psi1)
    ∂yypsi1 = similar(psi1)
    ∂ypsi2  = similar(psi2)
    ∂yypsi2 = similar(psi2)

    # compute y‑derivatives
    deriv_y_and_yy!(∂ypsi1, ∂yypsi1, copy(psi1), params.dy)
    deriv_y_and_yy!(∂ypsi2, ∂yypsi2, copy(psi2), params.dy)

    # precompute inv dx, etc.
    inv2dx = 1.0 / (2 * params.dx)
    invdx2 = 1.0 / (params.dx^2)

    # main loop over y, parallelized via threads
    Threads.@threads for j in 1:Ny
        # local accumulators
        local_Fh_col = zeros(Float64, ncbins)
        local_Fv_col = zeros(Float64, ncbins)

        # extract 2D slices
        ψ1j = psi1[:, j, :]
        ψ2j = psi2[:, j, :]

        # compute u1′ = -∂yψ1′
        u1j = - view(∂ypsi1, :, j, :)

        # compute v1′ = ∂xψ1′  (periodic central difference in x); should I do this spectrally?
        v1j = similar(ψ1j)
        @inbounds @turbo for t in 1:nt, i in 1:Nx
            # periodic indexing
            ip = (i == Nx ? 1 : i + 1)
            im = (i == 1 ? Nx : i - 1)
            v1j[i, t] = (ψ1j[ip, t] - ψ1j[im, t]) * inv2dx
        end

        # compute ζ1′ = ∂xx + ∂yy on ψ1′ for this j
        ζ1j = similar(ψ1j)
        # ∂xx part
        @inbounds @turbo for t in 1:nt, i in 1:Nx
            ip = (i == Nx ? 1 : i + 1)
            im = (i == 1 ? Nx : i - 1)
            ζ1j[i, t] = (ψ1j[ip, t] - 2.0 * ψ1j[i, t] + ψ1j[im, t]) * invdx2
        end
        # add ∂yy part
        @inbounds @turbo for t in 1:nt, i in 1:Nx
            ζ1j[i, t] += ∂yypsi1[i, j, t]
        end

        # define A/B for heat flux and vorticity flux
        A_heat = v1j .- mean(v1j, dims=1)
        B_heat = ψ2j .- ψ1j .- mean(ψ2j .- ψ1j, dims=1)
        A_vort = v1j .- mean(v1j, dims=1)
        B_vort = ζ1j .- mean(ζ1j, dims=1)


        # FFTs
        A_heat_ft = fft(A_heat)
        B_heat_ft = fft(B_heat)
        A_vort_ft = fft(A_vort)
        B_vort_ft = fft(B_vort)

        # binning over k, ω
        for ik in 1:Nx
            k = params.k_vec[ik]
            if k == 0.0
                continue
            end
            for iw in 1:nt
                ω = params.ω_vec[iw]
                c = ω / k
                # find bin
                b = searchsortedfirst(params.c_bin_edges, c) - 1    # find first bin >= local c
                if b < 1 || b > ncbins
                    continue
                end
                # cross spectrum contributions
                Sh = real( A_heat_ft[ik, iw] * conj(B_heat_ft[ik, iw]) )
                Sv = real( A_vort_ft[ik, iw] * conj(B_vort_ft[ik, iw]) )
                local_Fh_col[b] += Sh
                local_Fv_col[b] += Sv
            end
        end

        # normalization for this j
        # note: normalization is global (Nx*nt) but we can divide later
        # accumulate into global arrays
        @inbounds for b in 1:ncbins
            F_heat[b, j] += local_Fh_col[b]
            F_vort[b, j] += local_Fv_col[b]
        end
    end  # end Threads.@threads

    # Global normalization
    F_heat ./= (Nx * nt)
    F_vort ./= (Nx * nt)

    return F_heat, F_vort, params.c_bin_centers, -hcat(vec(∂yψ1m), vec(∂yyψ1m), vec(∂yψ2m), vec(∂yyψ2m))
end


"""
Sliding block version (optimized).

Arguments:
  psi1_all, psi2_all :: (Nx, Ny, nt_total)
  params :: SpectParams
  block_len, block_stride :: Int

Returns:
  average heat and vorticity spectra over all blocks
"""
function compute_flux_spectra_block(psi1_all::Array{Float64,3}, psi2_all::Array{Float64,3}, model_params, dx, dy, params::SpectParams, block_len::Int, block_stride::Int)
    Nx, Ny, nt_total = size(psi1_all)

    ncbins = params.ncbins

    sumFh = zeros(Float64, ncbins, Ny)
    sumFv = zeros(Float64, ncbins, Ny)
    count = 0

    t0 = 1
    while t0 + block_len - 1 ≤ nt_total
        t1 = t0 + block_len - 1
        psi1_block = @view psi1_all[:, :, t0:t1]
        psi2_block = @view psi2_all[:, :, t0:t1]
        # make a temp params for this block if needed (nt changed)
        # here we assume params.nt == block_len; if not, you can create new SpectParams
        if params.nt != block_len
            # recreate params with updated nt, ω_vec etc
            params_block = make_spectral_params_block(Nx, Ny, block_len,
                                               dx, dy, params.dt, maximum(abs.(params.c_bin_centers)), params.ncbins)
            # params_block = make_block_spectral_params(block_len, params, dx, dy, maximum(abs.(params.c_bin_centers)), params.ncbins)
        else
            params_block = params
        end

        Fh_block, Fv_block, c, Ubar = compute_phase_speed_spectra(psi1_block, psi2_block, params_block)

        sumFh .+= Fh_block
        sumFv .+= Fv_block
        count += 1
        t0 += block_stride
    end

    sumFh ./= count
    sumFv ./= count

    return sumFh, sumFv, params.c_bin_centers
end

# end  # module PhaseSpeedSpectraQG
