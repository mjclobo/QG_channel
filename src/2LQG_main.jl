

################################################################################
# Operations
################################################################################

function create_finite_difference_laplacian_y(Ny::Int, dy::Float64)
    """
    Create a finite difference Laplacian operator in the y-direction
    using second-order central differences.

    No boundary conditions are explicitly enforced.

    This operator acts on the y-dimension (columns) of an Nx×Ny matrix.
    """
    main_diag = fill(-2.0, Ny)
    off_diag = fill(1.0, Ny - 1)

    L = spdiagm(
        -1 => off_diag,
         0 => main_diag,
         1 => off_diag
    )

    return L / dy^2
end


# for integrating factor method; these linear operators are in kx-space
using SparseArrays, LinearAlgebra

function precompute_linear_operators(Nx, Ny, dx, dy, F1, F2)
    """
    Precomputes the linear evolution operators for the integrating factor method
    in a two-layer QG model with periodic BCs in x and solid walls in y.

    Returns:
        L_ops :: Vector of 2Ny × 2Ny matrices, one for each kx mode.
    """
    # Define wavenumbers in x (FFT convention)
    kx = 2π * vcat(0:floor(Int, Nx/2), -ceil(Int, (Nx-1)/2):-1) / (Nx * dx)
    k2 = kx.^2

    # Identity and Laplacian in y-direction (finite difference)
    Iy = I(Ny)
    Dy2 = create_finite_difference_laplacian_y(Ny, dy)  # FD Laplacian in y

    # Preallocate list of linear operators
    L_ops = Vector{Matrix{Float64}}(undef, length(kx))

    for i in 1:length(kx)
        kx2 = k2[i]

        # Total Laplacian: spectral in x, FD in y
        Lap_kx = -kx2 * Iy + Dy2

        # Build 2-layer linear operator matrix
        A11 = Lap_kx - F1 * Iy
        A12 = F1 * Iy
        A21 = F2 * Iy
        A22 = Lap_kx - F2 * Iy

        A = [A11 A12;
             A21 A22]

        # Compute matrix exponential for integrating factor
        L_ops[i] = exp(Matrix(dt * A))

    end

    return L_ops
end


# Precompute matrix and factorization; this is in x space
function precompute_operators(Nx, Ny)
    N = Nx * Ny
    L_op = build_sparse_laplacian_neumann_x(Nx, Ny, dx, dy)
    I_N = spdiagm(0 => ones(N))

    A11 = L_op - F1 * I_N
    A12 = F1 * I_N
    A21 = F2 * I_N
    A22 = L_op - F2 * I_N

    A = [A11 A12;
         A21 A22]

    # Factorize once
    F = lu(A)
    return F
end

function d_dy(f::Matrix{Float64}, dy::Float64)
    Nx, Ny = size(f)
    df = similar(f)

    @threads for i in 1:Nx
        @inbounds begin
            # Second-order forward difference at bottom
            df[i, 1] = (-3f[i, 1] + 4f[i, 2] - f[i, 3]) / (2dy)

            # Centered differences for interior
            @turbo for j in 2:Ny-1
                df[i, j] = (f[i, j+1] - f[i, j-1]) / (2dy)
            end

            # Second-order backward difference at top
            df[i, Ny] = (3f[i, Ny] - 4f[i, Ny-1] + f[i, Ny-2]) / (2dy)
        end
    end

    return df
end


function d_dx(f::AbstractMatrix{T}, dx::T) where T
    Nx, Ny = size(f)
    df = similar(f)

    dx2 = 2dx

    # Thread over y (columns)
    @threads for j in 1:Ny
        @inbounds begin
            # Periodic boundary at i = 1 and i = Nx
            df[1, j]    = (f[2, j]   - f[Nx, j]) / dx2
            df[Nx, j]   = (f[1, j]   - f[Nx-1, j]) / dx2

            # Vectorized interior loop
            @turbo for i in 2:Nx-1
                df[i, j] = (f[i+1, j] - f[i-1, j]) / dx2
            end
        end
    end

    return df
end


function laplacian_operator_neumann_x(Nx, Ny, dx, dy)
    # should I make x stuff spectral?
    dx2 = dx^2
    dy2 = dy^2

    function L(ψ)
        if size(ψ) != (Nx, Ny)
            error("ψ must be same size as (Nx, Ny) used to defin Laplacian operator")
        end

        lap = similar(ψ)

        @threads for j in 1:Ny
            @inbounds begin
                # Precompute Neumann boundaries in y
                jp = (j == Ny) ? Ny : j + 1
                jm = (j == 1)  ? 1  : j - 1

                @turbo for i in 1:Nx
                    ip = (i == Nx) ? 1 : i + 1  # Periodic east
                    im = (i == 1)  ? Nx : i - 1 # Periodic west

                    d2ψ_dx2 = (ψ[ip, j] - 2ψ[i, j] + ψ[im, j]) / dx2
                    d2ψ_dy2 = (ψ[i, jp] - 2ψ[i, j] + ψ[i, jm]) / dy2

                    lap[i, j] = d2ψ_dx2 + d2ψ_dy2
                end
            end
        end

        return lap
    end

    return L
end



function arakawa_jacobian(a, b)
    Nx, Ny = size(a)
    J = zeros(Float64, Nx, Ny)

    dx2 = 2dx
    dy2 = 2dy
    denom = 4dx * dy

    ip = [i == Nx ? 1 : i + 1 for i in 1:Nx]
    im = [i == 1  ? Nx : i - 1 for i in 1:Nx]
    jp = [j == Ny ? j - 1 : j + 1 for j in 1:Ny]
    jm = [j == 1  ? j + 1 : j - 1 for j in 1:Ny]

    @threads for i in 1:Nx
        i_p = ip[i]
        i_m = im[i]

        @inbounds @turbo for j in 1:Ny
            j_p = jp[j]
            j_m = jm[j]

            dady = (a[i, j_p] - a[i, j_m]) / dy2
            dbdx = (b[i_p, j] - b[i_m, j]) / dx2

            dadx = (a[i_p, j] - a[i_m, j]) / dx2
            dbdy = (b[i, j_p] - b[i, j_m]) / dy2

            J1 = dadx * dbdy - dady * dbdx

            J2 = (
                a[i_p, j] * (b[i_p, j_p] - b[i_p, j_m]) -
                a[i_m, j] * (b[i_m, j_p] - b[i_m, j_m]) -
                a[i, j_p] * (b[i_p, j_p] - b[i_m, j_p]) +
                a[i, j_m] * (b[i_p, j_m] - b[i_m, j_m])
            ) / denom

            J3 = (
                a[i_p, j_p] * b[i, j_p] - a[i_m, j_p] * b[i, j_p] -
                a[i_p, j_m] * b[i, j_m] + a[i_m, j_m] * b[i, j_m] -
                a[i_p, j_p] * b[i_p, j] + a[i_p, j_m] * b[i_p, j] +
                a[i_m, j_p] * b[i_m, j] - a[i_m, j_m] * b[i_m, j]
            ) / denom

            J[i, j] = (J1 + J2 + J3) / 3
        end
    end

    return J
end


function build_sparse_laplacian_neumann_x(Nx, Ny, dx, dy)
    N = Nx * Ny
    D = spzeros(N, N)

    idx = (i, j) -> (j - 1) * Nx + i

    for j in 1:Ny
        for i in 1:Nx
            n = idx(i, j)

            D[n, n] = -2 / dx^2 - 2 / dy^2

            # Periodic in x
            D[n, idx(mod1(i - 1, Nx), j)] += 1 / dx^2  # west
            D[n, idx(mod1(i + 1, Nx), j)] += 1 / dx^2  # east

            # Neumann in y
            if j > 1
                D[n, idx(i, j - 1)] += 1 / dy^2
            else
                D[n, n] += 1 / dy^2  # Neumann bottom
            end

            if j < Ny
                D[n, idx(i, j + 1)] += 1 / dy^2
            else
                D[n, n] += 1 / dy^2  # Neumann top
            end
        end
    end

    return D
end


################################################################################
# define operators used throughout
################################################################################

# function for Laplacian function, L2D(ψ)
L2D = laplacian_operator_neumann_x(Nx, Ny, dx, dy)

# matrix operator for QG PV inversion
inversion_ops = precompute_operators(Nx, Ny)

# if timestep_method=="RK4_int"
    # set of vector operators for integrating factor time stepping
    L_ops = precompute_linear_operators(Nx, Ny, dx, dy, F1, F2)
# end


################################################################################
#  2LQG main functions
################################################################################


function compute_qg_pv(ψ1::Array{Float64,2}, ψ2::Array{Float64,2}; lap_op=L2D)

    Nx, Ny = size(ψ1)

    q1 = zeros(Nx, Ny)
    q2 = zeros(Nx, Ny)

    # lap_ψ1 = L2D(ψ1)
    # lap_ψ2 = L2D(ψ2)

    # Compute PV; should I add beta here? Doesn't matter for most important thing, i.e., model integration
    q1 = lap_op(ψ1) .+ F1 .* (ψ2 .- ψ1) 
    q2 = lap_op(ψ2) .+ F2 .* (ψ1 .- ψ2)

    return q1, q2
end



function invert_qg_pv(q1, q2)
    Nx, Ny = size(q1)
    N = Nx * Ny

    # Vectorize inputs
    rhs = [vec(q1); vec(q2)]

    # Solve system
    ψ_vec = inversion_ops \ rhs

    # Reshape outputs
    ψ1 = reshape(ψ_vec[1:N], Nx, Ny)
    ψ2 = reshape(ψ_vec[N+1:end], Nx, Ny)

    # Remove mean to fix gauge freedom
    ψ̄ = mean((ψ1 .+ ψ2) ./ 2)
    ψ1 .-= ψ̄
    ψ2 .-= ψ̄

    return ψ1, ψ2
end


function rhs(q1, q2)
    ψ1, ψ2 = invert_qg_pv(q1, q2) #, k2, F, Ly_op) #, ops_dirichlet)

    J1 =  arakawa_jacobian(ψ1, q1) # zeros(Nx, Ny)  # 
    J2 =  arakawa_jacobian(ψ2, q2) # zeros(Nx, Ny)  # 

    dq1dt = -J1 .- beta .* u_from_psi(ψ1)[2]   # beta * v, where v = dψ/dx
    dq2dt = -J2 .- beta .* u_from_psi(ψ2)[2]

    dq1dt .-= ν .* L2D(L2D(q1))
    dq2dt .-= ν .* L2D(L2D(q2))
    dq2dt .-= r .* L2D(ψ2)

    # Thermal damping toward background shear
    dq1dt .+=  α * F1 .* ((ψ1 .- ψ2) .- ψ_diff_bg)
    dq2dt .+= -α * F2 .* ((ψ1 .- ψ2) .- ψ_diff_bg)

    return dq1dt, dq2dt
end



function u_from_psi(ψ)
    u = -d_dy(ψ, dy)

    ψ_hat = rfft(ψ, 1)                # (Nx/2+1, Ny)

    # KXr2d = repeat(KXr, 1, Ny)  # (Nx/2+1, Ny)

    v_hat = 1im .* KXr .* ψ_hat       # dψ/dx in spectral space
    v = irfft(v_hat, Nx, 1) |> real   # back to x space

    return u, v
end



################################################################################
#  For zonal-mean zonal momentum equation
################################################################################


function diagnose_zonal_mean_meridional_velocity(
    ψ1::Array{Float64,2}, ψ2::Array{Float64,2},
    dx::Float64, dy::Float64,
    beta::Float64, f0::Float64, gprime::Float64,
    H1, H2, lap_yband)

    Nx, Ny = size(ψ1)
 
    u1, v1 = u_from_psi(ψ1)
    u2, v2 = u_from_psi(ψ2)

    # Compute PV in each layer: q = ∇²ψ + F(ψ2 - ψ1) + βy
    F1 = (f0^2) / (gprime * H1)  # coupling coefficient upper
    F2 = (f0^2) / (gprime * H2)  # coupling coefficient lower

    y = dy * (0:Ny-1)
    betay = reshape(beta .* y, 1, Ny)  # make it broadcastable

    q1 = lap_yband(ψ1) .+ F1 .* (ψ2 .- ψ1) .+ betay
    q2 = lap_yband(ψ2) .+ F2 .* (ψ1 .- ψ2) .+ betay

    # Compute eddy PV fluxes: v'q' = v*q - mean(v)*mean(q)
    # But mean(v) = 0, so just zonal mean of v*q

    vq1 = v1 .* q1
    vq2 = v2 .* q2

    vq1_bar = mean(vq1, dims=1)
    vq2_bar = mean(vq2, dims=1)

    # Compute divergence of eddy PV flux (∂/∂y)

    div_vq = d_dy(vq1_bar, dy)  # vq1_bar = -vq2_bar, so only need one

    # Invert for overturning streamfunction Ψ(y):
    # dΨ/dy = -v̄1 = from inversion of eddy PV flux divergence
    # So integrate: Ψ(y) = ∫ -v̄1 dy = ∫ ∂(v'q')/∂y dy ⇒ -v̄1 = -div_vq ⇒ v̄1 = div_vq

    Ψ = -cumsum(div_vq .* dy; dims=2)

    # Compute mean meridional velocities
    vbar1 = -diff(Ψ, dims=2) ./ dy
    vbar2 = +diff(Ψ, dims=2) ./ dy

    # Pad to original size (Ny-1 → Ny)
    function pad_y(field)
        cat(field, field[:,end:end]; dims=2)
    end
    vbar1 = pad_y(vbar1)
    vbar2 = pad_y(vbar2)

    return vbar1, vbar2, Ψ
end

function diagnose_meridional_velocity_from_omega_nonperiodic_y(
    ψ1::Array{Float64,2}, ψ2::Array{Float64,2},
    dx::Float64, dy::Float64)

    Nx, Ny = size(ψ1)

    # --- 1. Define derivatives (central diff. in y, periodic in x) ---
    function ddx(field)
        (circshift(field, (-1, 0)) .- circshift(field, (1, 0))) ./ (2dx)
    end

    function ddy(field)
        out = similar(field)
        out[:, 2:Ny-1] = (field[:, 3:Ny] .- field[:, 1:Ny-2]) ./ (2dy)
        # One-sided differences at boundaries (2nd order)
        out[:, 1] = (field[:, 2] .- field[:, 1]) ./ dy
        out[:, end] = (field[:, end] .- field[:, end-1]) ./ dy
        return out
    end

    function d2dy2(field)
        out = similar(field)
        out[:, 2:Ny-1] = (field[:, 3:Ny] .- 2 .* field[:, 2:Ny-1] .+ field[:, 1:Ny-2]) ./ dy^2
        # Dirichlet BCs: ω = 0 at y boundaries ⇒ field = 0 outside domain
        out[:, 1] = (field[:, 2] .- 2 .* field[:, 1]) ./ dy^2
        out[:, end] = (field[:, end-1] .- 2 .* field[:, end]) ./ dy^2
        return out
    end

    # --- 2. Compute geostrophic velocities and temperature gradients ---
    u1 = -ddy(ψ1); v1 = ddx(ψ1)
    u2 = -ddy(ψ2); v2 = ddx(ψ2)

    T = ψ1 .- ψ2
    Tx = ddx(T)
    Ty = ddy(T)

    # --- 3. Q-vector components ---
    function compute_Q(u, v)
        ux = ddx(u); uy = ddy(u)
        vx = ddx(v); vy = ddy(v)

        Qx = - (ux .* Tx .+ vx .* Ty)
        Qy = - (uy .* Tx .+ vy .* Ty)
        return Qx, Qy
    end

    Qx1, Qy1 = compute_Q(u1, v1)
    Qx2, Qy2 = compute_Q(u2, v2)

    Qx = 0.5 .* (Qx1 .+ Qx2)
    Qy = 0.5 .* (Qy1 .+ Qy2)

    divQ = ddx(Qx) .+ ddy(Qy)  # ∇·Q

    # --- 4. Solve Poisson equation: ∇²ω = -2 ∇·Q
    # Use FFT in x, finite difference in y
    rhs = -2 .* divQ

    # Prepare output
    ω = zeros(Nx, Ny)

    # FFT wave numbers
    kx = [2π * (i <= Nx ÷ 2 ? i - 1 : i - Nx - 1) / (Nx * dx) for i in 1:Nx]

    # Solve for each x-wavenumber separately
    for (ix, k) in enumerate(kx)
        λx = -(k^2)  # spectral Laplacian in x

        # Build tridiagonal matrix for y-diff Laplacian: d²/dy² + λx
        main_diag = fill(-2/dy^2 + λx, Ny)
        off_diag = fill(1/dy^2, Ny-1)

        # Apply Dirichlet BCs: ω = 0 at y = 0, L ⇒ modify matrix
        A = Tridiagonal(off_diag, main_diag, off_diag)

        rhs_slice = rhs[ix, :]

        # Solve: A * ω_hat = rhs ⇒ ω(x_k, y)
        ω[ix, :] = A \ rhs_slice
    end

    # --- 5. Compute zonal-mean meridional velocities via mass continuity ---
    # ∂v_ag/∂y + ∂ω/∂z = 0 ⇒ in two-layer: v̄1 = -ω, v̄2 = +ω (scaled by Δp = 1)

    vbar1 = -ω
    vbar2 = +ω

    # Zonal averages (mean over x)
    vbar1_zonal = mean(vbar1, dims=1)
    vbar2_zonal = mean(vbar2, dims=1)

    return vbar1_zonal[:], vbar2_zonal[:], ω
end
