

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
    idx = (i, j) -> (j - 1) * Nx + i

    # L_op = build_sparse_laplacian_neumann_x(Nx, Ny, dx, dy)
    L_op = build_sparse_laplacian_dirichlet_y(Nx, Ny, dx, dy)
    I_N = spdiagm(0 => ones(N))

    A11 = L_op - F1 * I_N
    A12 = F1 * I_N
    A21 = F2 * I_N
    A22 = L_op - F2 * I_N

    for j in (1, Ny)
        for i in 1:Nx
            n = idx(i, j)
            A11[n, :] .= 0.0
            A11[n, n] = 1.0
            A12[n, :] .= 0.0  # remove coupling
            
            A22[n, :] .= 0.0
            A22[n, n] = 1.0
            A21[n, :] .= 0.0  # remove coupling
        end
    end    

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

function laplacian_operator_free_slip_y(Nx, Ny, dx, dy)
    dx2 = dx^2
    dy2 = dy^2

    function L(ψ)
        if size(ψ) != (Nx, Ny)
            error("ψ must be size (Nx, Ny)")
        end

        lap = zeros(Nx, Ny)

        @threads for j in 2:Ny-1
            @inbounds begin
                jp = j + 1
                jm = j - 1
                @turbo for i in 1:Nx
                    ip = (i == Nx) ? 1 : i + 1
                    im = (i == 1)  ? Nx : i - 1

                    d2ψ_dx2 = (ψ[ip, j] - 2ψ[i, j] + ψ[im, j]) / dx2
                    d2ψ_dy2 = (ψ[i, jp] - 2ψ[i, j] + ψ[i, jm]) / dy2

                    lap[i, j] = d2ψ_dx2 + d2ψ_dy2
                end
            end
        end

        # # Enforce ∂y(∇²ψ)=0 at meridional walls (free-slip)
        lap[:, 1]  .= lap[:, 2]
        lap[:, Ny] .= lap[:, Ny-1]

        # lap[:, 1]  .= 0.0
        # lap[:, Ny] .= 0.0

        return lap
    end

    return L
end



function arakawa_jacobian(a, b, dx, dy)
    Nx, Ny = size(a)
    J = zeros(Float64, Nx, Ny)

    dx2 = 2dx
    dy2 = 2dy
    denom = 4dx * dy

    ip = [i == Nx ? 1 : i + 1 for i in 1:Nx]
    im = [i == 1  ? Nx : i - 1 for i in 1:Nx]

    @threads for i in 1:Nx
        i_p = ip[i]
        i_m = im[i]

        @inbounds for j in 2:Ny-1  # only interior rows
            j_p = j + 1
            j_m = j - 1

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

function arakawa_jacobian_doubly_periodic(a, b, dx, dy)
    Nx, Ny = size(a)
    J = zeros(Float64, Nx, Ny)

    dx2 = 2dx
    dy2 = 2dy
    denom = 4dx * dy

    # Periodic indices in x
    ip = [i == Nx ? 1 : i + 1 for i in 1:Nx]
    im = [i == 1  ? Nx : i - 1 for i in 1:Nx]

    # Periodic indices in y
    jp = [j == Ny ? 1 : j + 1 for j in 1:Ny]
    jm = [j == 1  ? Ny : j - 1 for j in 1:Ny]

    @threads for i in 1:Nx
        i_p = ip[i]
        i_m = im[i]

        @inbounds for j in 1:Ny
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

function BT_jacobian(ah, b, kx, ky)
    # J(f,g)=∂y​[(∂x​f)g]−∂x​[(∂y​f)g]

    ∂xa = real.(ifft(im .* kx .* ah))
    ∂ya = real.(ifft(im .* ky .* ah))

    return im .* ky .* fft(∂xa .* b) .- im .* kx .* fft(∂ya .* b)
end


function build_sparse_laplacian_dirichlet_y(Nx, Ny, dx, dy)
    N = Nx * Ny
    D = spzeros(N, N)

    idx = (i, j) -> (j - 1) * Nx + i

    for j in 1:Ny
        for i in 1:Nx
            n = idx(i, j)

            # Dirichlet in y means boundary rows must be modified later (see below)
            if j == 1 || j == Ny
                D[n, n] = 1.0  # placeholder, will be overridden for Dirichlet
                continue
            end

            D[n, n] = -2 / dx^2 - 2 / dy^2

            # Periodic in x
            D[n, idx(mod1(i - 1, Nx), j)] += 1 / dx^2  # west
            D[n, idx(mod1(i + 1, Nx), j)] += 1 / dx^2  # east

            # Interior points in y
            D[n, idx(i, j - 1)] += 1 / dy^2  # south
            D[n, idx(i, j + 1)] += 1 / dy^2  # north
        end
    end

    return D
end


function biharmonic(q; L=L2D) # q or psi
    lap1 = L(q)

    lap1[:, 1] .= lap1[:, 2]        # ∂(∇²ψ)/∂y = 0 at y = 0
    lap1[:, end] .= lap1[:, end-1]  # ∂(∇²ψ)/∂y = 0 at y = Ly

    return L(lap1)
end

function hyperviscous(ψ; L=L2D)
    lap1 = L(ψ)          # ∇²ψ
    lap2 = L(lap1)       # ∇⁴ψ

    # Enforce ∂y(∇⁴ψ) = 0 at y = 0, Ly
    lap2[:, 1]  .= lap2[:, 2]
    lap2[:, end] .= lap2[:, end-1]

    return L(lap2)  # ∇⁶ψ = L(∇⁴ψ)
end


################################################################################
# define operators used throughout
################################################################################

if Nz>1
    # function for Laplacian function, L2D(ψ)
    L2D = laplacian_operator_free_slip_y(Nx, Ny, dx, dy)


    # matrix operator for QG PV inversion
    inversion_ops = precompute_operators(Nx, Ny)

# if timestep_method=="RK4_int"
    # set of vector operators for integrating factor time stepping
    L_ops = precompute_linear_operators(Nx, Ny, dx, dy, F1, F2)
# end
end

################################################################################
#  2LQG main functions
################################################################################


function compute_qg_pv(ψ1::Array{Float64,2}, ψ2::Array{Float64,2}; lap_op=L2D)

    Nx, Ny = size(ψ1)

    q1 = zeros(Nx, Ny)
    q2 = zeros(Nx, Ny)

    # Compute PV; note that we don't add beta, as this makes PV inversion inconsistent
    q1 = lap_op(ψ1) .+ F1 .* (ψ2 .- ψ1) # 
    q2 = lap_op(ψ2) .+ F2 .* (ψ1 .- ψ2)  # 

    return q1, q2
end

function invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy)
    Nx, Ny = size(q1)
    N = Nx * Ny

    idx = (i, j) -> (j - 1) * Nx + i

    # Vectorize q and apply Dirichlet BC adjustments
    rhs1 = copy(vec(q1))
    rhs2 = copy(vec(q2))

    for j in (1, Ny)  # Top and bottom boundaries
        for i in 1:Nx
            n = idx(i, j)

            rhs1[n] = ψ1_bg[i, j]
            rhs2[n] = ψ2_bg[i, j]
        end
    end

    rhs = [rhs1; rhs2]

    # Solve system
    ψ_vec = inversion_ops \ rhs

    # Reshape
    ψ1 = reshape(ψ_vec[1:N], Nx, Ny)
    ψ2 = reshape(ψ_vec[N+1:end], Nx, Ny)

    # # Enforce ψ = ψ_bg at top/bottom
    ψ1[:, 1] .= ψ1_bg[:, 1]
    ψ1[:, end] .= ψ1_bg[:, end]

    ψ2[:, 1] .= ψ2_bg[:, 1]
    ψ2[:, end] .= ψ2_bg[:, end]

    return ψ1, ψ2
end


function rhs(q1, q2)
    ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, inversion_ops, dx, dy) # invert_qg_pv(q1, q2)

    J1 =  arakawa_jacobian(ψ1, q1, dx, dy) # zeros(Nx, Ny)  # 
    J2 =  arakawa_jacobian(ψ2, q2, dx, dy) # zeros(Nx, Ny)  # 

    dq1dt = -J1 .- beta .* u_from_psi(ψ1)[2]   # beta * v, where v = dψ/dx
    dq2dt = -J2 .- beta .* u_from_psi(ψ2)[2]

    # dq1dt .-= ν .* biharmonic(q1 .- q1_bg)
    # dq2dt .-= ν .* biharmonic(q2 .- q2_bg)

    # dq1dt .-= ν .* biharmonic(L2D(ψ1))
    # dq2dt .-= ν .* biharmonic(L2D(ψ2))

    dq1dt .-= ν .* hyperviscous(ψ1)
    dq2dt .-= ν .* hyperviscous(ψ2)
    
    dq2dt .-= r .* L2D(ψ2)

    # Thermal damping toward background shear
    dq1dt .+=  α * F1 .* ((ψ1 .- ψ2) .- ψ_diff_bg)
    dq2dt .+= -α * F2 .* ((ψ1 .- ψ2) .- ψ_diff_bg)

    return dq1dt, dq2dt
end


# ψf = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

# ψfh = fft(ψf)

# topo_forc = @. -(-k2D - Ld^-2) * ψfh

# # qh .+= topo_forc

# q = real.(ifft(qh))

# ψh = @. (-k2D - Ld^-2) * qh

# dqhdt = - BT_jacobian(ψh, q, kx, ky) # zeros(Nx, Ny)  # J(f,g)=∂y​[(∂x​f)g]−∂x​[(∂y​f)g]

# # dqhdt .-= beta * im * kx .* ψh

# dqhdt .+= BT_jacobian(ψh, real.(ifft(topo_forc)), kx, ky)

## different method

"""
    conservative_derivs(f, dx, dy)

Compute conservative first and second derivatives of 2D array `f` on a uniform grid.

Returns:
    fx  - ∂f/∂x (conservative, periodic)
    fy  - ∂f/∂y (conservative, periodic)
    lapf - ∇² f = ∂²f/∂x² + ∂²f/∂y²
"""
function ddx_ddy_reentrant_FD(f::Array{Float64,2}, dx::Float64, dy::Float64)
    Nx, Ny = size(f)
    
    # First derivatives (central differences, flux-conservative)
    fx = zeros(Nx, Ny)
    fy = zeros(Nx, Ny)
    
    # ∂/∂x
    for j in 1:Ny
        for i in 1:Nx
            ip = i < Nx ? i+1 : 1   # periodic
            im = i > 1  ? i-1 : Nx
            fx[i,j] = (f[ip,j] - f[im,j]) / (2*dx)
        end
    end
    
    # ∂/∂y
    for i in 1:Nx
        for j in 1:Ny
            jp = j < Ny ? j+1 : 1
            jm = j > 1  ? j-1 : Ny
            fy[i,j] = (f[i,jp] - f[i,jm]) / (2*dy)
        end
    end
    
    # # Second derivatives (Laplacian)
    # lapf = zeros(Nx, Ny)
    # for j in 1:Ny
    #     jp = j < Ny ? j+1 : 1
    #     jm = j > 1  ? j-1 : Ny
    #     for i in 1:Nx
    #         ip = i < Nx ? i+1 : 1
    #         im = i > 1  ? i-1 : Nx
    #         lapf[i,j] = (f[ip,j] - 2f[i,j] + f[im,j]) / dx^2 +
    #                     (f[i,jp] - 2f[i,j] + f[i,jm]) / dy^2
    #     end
    # end
    
    return dfdx, dfdy
end


function wavemaker_m1(ψ0, x, y, x0, y0, R, δ, t, τ)
    # x: 1D array of length Nx
    # y: 1D array of length Ny
    # returns Nx x Ny array

    Nx = length(x)
    Ny = length(y)

    # Make 2D meshgrid arrays
    X = repeat(x, 1, Ny)          # Nx x Ny, each row is x
    Y = repeat(y', Nx, 1)         # Nx x Ny, each column is y

    # Shift coordinates to patch center
    Xc = X .- x0
    Yc = Y .- y0

    # Radius and azimuth
    r = @. sqrt(Xc^2 + Yc^2)
    θ = @. atan(Yc, Xc)

    # m=1 azimuthal wavemaker at patch edge
    ψ_f = @. ψ0 * sin(2*pi*t/τ) * r * cos(θ) * exp(-((r - R)^2 / (2*δ^2)))

    return ψ_f
end


function define_reentrant_lap_op_FD(Nx, Ny, dx, dy; inv_op=true)
    N = Nx*Ny
    
    dx2 = dx^2
    dy2 = dy^2

    # function to convert (i,j) to linear index
    idx(i,j) = (j-1)*Nx + i

    # build sparse matrix
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for j in 1:Ny
        jp = j < Ny ? j+1 : 1
        jm = j > 1 ? j-1 : Ny
        for i in 1:Nx
            ip = i < Nx ? i+1 : 1
            im = i > 1 ? i-1 : Nx

            center = idx(i,j)

            push!(rows, center); push!(cols, center); push!(vals, -2/dx2 - 2/dy2 - 1/Ld^2)
            push!(rows, center); push!(cols, idx(ip,j)); push!(vals, 1/dx2)
            push!(rows, center); push!(cols, idx(im,j)); push!(vals, 1/dx2)
            push!(rows, center); push!(cols, idx(i,jp)); push!(vals, 1/dy2)
            push!(rows, center); push!(cols, idx(i,jm)); push!(vals, 1/dy2)

        end
    end

    L = sparse(rows, cols, vals, N, N)
    if inv_op==true
        L[1,:] .= 1.0
    end

    return L
end
function define_laplacian_FD(Nx, Ny, dx, dy)
    dx2 = dx^2
    dy2 = dy^2

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    idx(i,j) = (j-1)*Nx + i

    for j in 1:Ny
        jp = (j == Ny) ? 1 : j+1
        jm = (j == 1)  ? Ny : j-1
        for i in 1:Nx
            ip = (i == Nx) ? 1 : i+1
            im = (i == 1)  ? Nx : i-1

            center = idx(i,j)

            push!(rows, center); push!(cols, center); push!(vals, -2/dx2 -2/dy2)
            push!(rows, center); push!(cols, idx(ip,j)); push!(vals, 1/dx2)
            push!(rows, center); push!(cols, idx(im,j)); push!(vals, 1/dx2)
            push!(rows, center); push!(cols, idx(i,jp)); push!(vals, 1/dy2)
            push!(rows, center); push!(cols, idx(i,jm)); push!(vals, 1/dy2)
        end
    end

    return sparse(rows, cols, vals, Nx*Ny, Nx*Ny)
end


# define ∇^2 - Ld^-2 operator matrix
L_re_FD_inv = define_reentrant_lap_op_FD(Nx, Ny, dx, dy)
L_re_FD     = define_reentrant_lap_op_FD(Nx, Ny, dx, dy; inv_op=false)
Lap_op      = define_laplacian_FD(Nx, Ny, dx, dy)


function rhs_BT_FD(q, t)

    Q = q .- f0
    Q .-= mean(Q)           # ensure zero mean
    Q_vec = reshape(Q, Nx*Ny)
    ψ_v = reshape(L_re_FD_inv \ Q_vec, Nx, Ny)

    ψ_f = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)
    # ψ_f .-= mean(ψ_f)
    ψ_f_vec = reshape(ψ_f, Nx*Ny)

    q_f = -reshape(L_re_FD * ψ_f_vec, Nx, Ny)

    dqdt = - arakawa_jacobian_doubly_periodic(ψ_v .+ ψ_f, q .+ q_f, dx, dy)

    dqdt .-= r .* q

    return dqdt

end

function rhs(qh, t)

    q = real.(ifft(qh))
    Qh = fft(q .- f0)
    ψ_vh = @. -(k2D + Ld^-2).^-1 * Qh

    ψ_f = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

    # ψ_f = wavemaker_m1(ψ0, x, y, x0, y0, radius, δx, t, τ)

    dqhdt = - BT_jacobian(ψ_vh .+ fft(ψ_f), q .+ real.(ifft((k2D .+ Ld^-2) .* fft(ψ_f))), kx, ky)

    dqhdt .-= ν .* k2D .* k2D .* qh
    
    dqhdt .-= r .* qh

    return dqhdt
end

# function rhs(qh, t)
#     q = real.(ifft(qh))
#     Qh = fft(q .- f0)
#     ψ_vh = @. -(k2D + Ld^-2).^-1 * Qh

#     ψ_f = wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

#     # PV forcing from wavemaker
#     q_forcing = (k2D .+ Ld^-2) .* fft(ψ_f)

#     # Advection of existing PV
#     dqhdt = - BT_jacobian(ψ_vh, q, kx, ky)

#     # Add forcing
#     dqhdt .+= q_forcing

#     # Dissipation
#     dqhdt .-= ν .* k2D .* k2D .* qh
#     dqhdt .-= r .* qh

#     return dqhdt
# end



function u_from_psi(ψ)
    u = -d_dy(ψ, dy)

    ψ_hat = rfft(ψ, 1)                # (Nx/2+1, Ny)

    v_hat = 1im .* KXr .* ψ_hat       # dψ/dx in spectral space
    v = irfft(v_hat, Nx, 1) |> real   # back to x space

    return u, v
end


function u_from_psi_fft(ψh, kx, ky)
    return -real.(ifft(im .* ky .* ψh)), real.(ifft(im .* kx .* ψh))
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


function diagnose_meridional_velocity_from_omega_nonperiodic_y_boussinesq(
    ψ1::Array{Float64,2}, ψ2::Array{Float64,2},
    dx::Float64, dy::Float64,
    H1::Float64, H2::Float64
)

    Nx, Ny = size(ψ1)

    # --- Finite differences in y, periodic in x ---
    function ddx(field)
        (circshift(field, (-1, 0)) .- circshift(field, (1, 0))) ./ (2dx)
    end

    function ddy(field)
        out = similar(field)
        out[:, 2:Ny-1] .= (field[:, 3:Ny] .- field[:, 1:Ny-2]) ./ (2dy)
        out[:, 1] .= (field[:, 2] .- field[:, 1]) ./ dy
        out[:, end] .= (field[:, end] .- field[:, end-1]) ./ dy
        return out
    end

    function d2dy2(field)
        out = similar(field)
        out[:, 2:Ny-1] .= (field[:, 3:Ny] .- 2 .* field[:, 2:Ny-1] .+ field[:, 1:Ny-2]) ./ dy^2
        out[:, 1] .= (field[:, 2] .- 2 .* field[:, 1]) ./ dy^2
        out[:, end] .= (field[:, end-1] .- 2 .* field[:, end]) ./ dy^2
        return out
    end

    # --- Geostrophic velocities ---
    u1 = -ddy(ψ1); v1 = ddx(ψ1)
    u2 = -ddy(ψ2); v2 = ddx(ψ2)

    # --- Temperature gradient ~ interface displacement ~ ψ1 - ψ2
    T = ψ1 .- ψ2
    Tx = ddx(T)
    Ty = ddy(T)

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

    divQ = ddx(Qx) .+ ddy(Qy)

    # --- Solve Poisson equation: ∇² w = -2 ∇·Q ---
    rhs = -2 .* divQ
    w = zeros(Nx, Ny)

    # FFT wave numbers in x
    kx = [2π * (i <= Nx ÷ 2 ? i - 1 : i - Nx - 1) / (Nx * dx) for i in 1:Nx]

    for (ix, k) in enumerate(kx)
        λx = -(k^2)

        main_diag = fill(-2/dy^2 + λx, Ny)
        off_diag = fill(1/dy^2, Ny - 1)
        A = Tridiagonal(off_diag, main_diag, off_diag)

        rhs_slice = rhs[ix, :]
        w[ix, :] = A \ rhs_slice
    end

    # --- Ageostrophic meridional velocities ---
    vbar1 = -w ./ H1
    vbar2 = +w ./ H2

    # Zonal means
    vbar1_zonal = mean(vbar1, dims=1)
    vbar2_zonal = mean(vbar2, dims=1)

    return vbar1_zonal[:], vbar2_zonal[:], w
end

# function enforce_no_normal_flow!(ψ)
#     ψ .-= mean(ψ[:, 1])     # bottom wall: constant ψ in x; hold at zero
#     ψ[:, end] .= mean(ψ[:, end]) # top wall: constant ψ in x
# end

# function apply_bc_with_background!(ψ1, ψ2, ψ1_bg, ψ2_bg)
#     # Dirichlet BC at walls (no normal flow)
#     ψ1[:, 1] .= ψ1_bg[1]
#     ψ1[:, end] .= ψ1_bg[end]

#     ψ2[:, 1] .= ψ2_bg[1]
#     ψ2[:, end] .= ψ2_bg[end]

#     # No-stress BC (match curvature with background)
#     ψ1[:, 2] .= 2ψ1_bg[2] - ψ1_bg[1]
#     ψ1[:, end-1] .= 2ψ1_bg[end-1] - ψ1_bg[end]

#     ψ2[:, 2] .= 2ψ2_bg[2] - ψ2_bg[1]
#     ψ2[:, end-1] .= 2ψ2_bg[end-1] - ψ2_bg[end]
# end


function init_solver(qh0, kx, ky, k2D, Ld, ψ0, x, y, x0, y0, δx, δy)

    Nx, Ny = size(qh0)

    # -------- FFT plans --------
    fft_plan  = plan_fft(qh0;  flags=FFTW.MEASURE)
    ifft_plan = plan_ifft(qh0; flags=FFTW.MEASURE)

    # -------- Spectral-space complex arrays --------
    q       = similar(qh0)             # will store ifft(qh)
    Qh      = similar(qh0)
    ψ_vh    = similar(qh0)
    tmph    = similar(qh0)
    tmp_c   = similar(qh0)             # complex buffer for FFT inputs
    tmp2_c  = similar(qh0)             # complex buffer for Jacobian FFTs
    J       = similar(qh0)

    # -------- Physical-space real arrays --------
    ψ_f     = similar(qh0, Float64)
    tmp_r   = similar(qh0, Float64)
    tmp2_r  = similar(qh0, Float64)

    # -------- Precompute Gaussian wavemaker spatial shape --------
    G = @. exp( -((x - x0)^2 / (2δx^2)) - ((y - y0)^2 / (2δy^2)) )

    # -------- Precompute K = k2D + Ld^-2 --------
    K  = @. k2D + Ld^-2
    ikx = im .* kx
    iky = im .* ky

    return (
        fft_plan  = fft_plan,
        ifft_plan = ifft_plan,

        # complex fields
        q       = q,
        Qh      = Qh,
        ψ_vh    = ψ_vh,
        tmph    = tmph,
        tmp_c   = tmp_c,
        tmp2_c  = tmp2_c,
        J       = J,

        # real fields
        ψ_f     = ψ_f,
        tmp_r   = tmp_r,
        tmp2_r  = tmp2_r,

        # constants
        G       = G,
        K       = K,
        ikx     = ikx,
        iky     = iky,
        ψ0      = ψ0
    )
end

# ================================================================
# ==================== WAVEMAKER =================================
# ================================================================
function wavemaker!(ψ_f, ψ0, G, t, τ)
    s = sin(2π * t / τ)
    @. ψ_f = ψ0 * G * s
end

# ================================================================
# ==================== JACOBIAN ==================================
# ================================================================
function BT_jacobian!(
    J, ah, b,
    ikx, iky,
    fft_plan, ifft_plan,
    tmp_r, tmp2_r, tmph, tmp2_c
)
    # ∂x a
    @. tmph = ikx * ah
    mul!(tmp2_c, ifft_plan, tmph)
    @. tmp_r = real(tmp2_c)

    # FFT(∂x a * b)
    @. tmp2_r = tmp_r * b
    @. tmp2_c = complex(tmp2_r)
    mul!(tmph, fft_plan, tmp2_c)
    @. tmph = iky * tmph

    # ∂y a
    @. tmph = iky * ah
    mul!(tmp2_c, ifft_plan, tmph)
    @. tmp_r = real(tmp2_c)

    # FFT(∂y a * b)
    @. tmp2_r = tmp_r * b
    @. tmp2_c = complex(tmp2_r)      # keep tmp2_c unchanged
    mul!(tmph, fft_plan, tmp2_c)     # tmph = FFT(tmp2_c)
    @. tmph = ikx * tmph             # multiply by ikx

    # Combine
    @. J = tmph - tmp2_c
end

# ================================================================
# ==================== RHS FUNCTION ==============================
# ================================================================


function rhs!(
    dqhdt, qh, t,
    ν, r, τ, f0, kx, ky,
    params
)
    @unpack fft_plan, ifft_plan,
            q, Qh, ψ_vh, ψ_f,
            tmph, tmp_c, tmp2_c, J,
            tmp_r, tmp2_r,
            G, K, ikx, iky, ψ0 = params

    # ------------------------------------------------
    # 1. q = real(ifft(qh))
    # ------------------------------------------------
    mul!(q, ifft_plan, qh)
    @. q = real(q)

    # ------------------------------------------------
    # 2. Qh = fft(q - f0)
    # ------------------------------------------------
    @. tmp_r = q - f0
    @. tmp_c = complex(tmp_r)
    mul!(Qh, fft_plan, tmp_c)

    # ------------------------------------------------
    # 3. ψ_vh = -Qh / K
    # ------------------------------------------------
    @. ψ_vh = -Qh / K

    # ------------------------------------------------
    # 4. Wavemaker forcing ψ_f (physical space)
    # ------------------------------------------------
    wavemaker!(ψ_f, ψ0, G, t, τ)  # writes directly into ψ_f

    # ------------------------------------------------
    # 5. Spectral field for Jacobian first argument: ψ_vh + FFT(ψ_f)
    # ------------------------------------------------
    @. tmp_c = ψ_f
    mul!(tmph, fft_plan, tmp_c)    # tmph = FFT(ψ_f)
    @. tmph += ψ_vh                # spectral field

    # ------------------------------------------------
    # 6. Physical-space field for Jacobian second argument: q + ifft(K * FFT(ψ_f))
    # ------------------------------------------------
    @. tmp_c = ψ_f
    mul!(tmp2_c, fft_plan, tmp_c)  # tmp2_c = FFT(ψ_f)
    @. tmp2_c = K .* tmp2_c         # multiply by K
    mul!(tmp_c, ifft_plan, tmp2_c) # tmp_c = ifft(K * FFT(ψ_f))
    @. tmp2_r = q + real.(tmp_c)   # second argument b

    # ------------------------------------------------
    # 7. Nonlinear Jacobian
    # ------------------------------------------------
    BT_jacobian!(J, tmph, tmp2_r,
                 ikx, iky,
                 fft_plan, ifft_plan,
                 tmp_r, tmp2_r, tmph, tmp2_c)

    @. dqhdt = -J

    # ------------------------------------------------
    # 8. Linear terms
    # ------------------------------------------------
    # viscosity (optional, precompute k4D if needed)
    # @. dqhdt -= ν * k4D * qh

    # linear drag
    @. dqhdt -= r * qh

    return nothing
end


######################################################################
## Below this line are functions for an optimized and
## GPU-compatible version of the FD BT model
######################################################################

struct BTParamsOpt
    Nx::Int
    Ny::Int
    dx::Float64
    dy::Float64
    r::Float64
    f0::Float64
    L_fac          # pre-factorized Laplacian operator
    L_op::SparseMatrixCSC{Float64,Int}
    Lap_op
    ip::Vector{Int}
    im::Vector{Int}
    jp::Vector{Int}
    jm::Vector{Int}
    backend     # KA backend, e.g. CPU() or CUDABackend()
end

@kernel function arakawa_kernel!(
    J, a, b, ip, im, jp, jm, dx, dy
)
    Nx, Ny = size(a)
    dx2 = 2dx
    dy2 = 2dy
    denom = 4dx * dy

    I = @index(Global, Linear)
    total = Nx * Ny

    # Guard without return
    if I <= total
        # convert linear index to (i,j)
        j = (I - 1) ÷ Nx + 1
        i = I - (j - 1) * Nx

        i_p = ip[i];  i_m = im[i]
        j_p = jp[j];  j_m = jm[j]

        #### Central derivatives
        dady = (a[i, j_p] - a[i, j_m]) / dy2
        dbdx = (b[i_p, j] - b[i_m, j]) / dx2

        dadx = (a[i_p, j] - a[i_m, j]) / dx2
        dbdy = (b[i, j_p] - b[i, j_m]) / dy2

        J1 = dadx * dbdy - dady * dbdx

        #### Arakawa terms
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


function rhs_BT_FD!(
    dqdt, q,
    ψ_v, ψ_v_vec,
    Q, Q_vec,
    ψ_f, ψ_f_vec, q_f, ψ_sum, q_sum,
    lap_q_vec, lap2_q_vec,
    params::BTParamsOpt, t
)

    @unpack Nx, Ny, dx, dy, r, f0, L_fac, L_op, ip, im, jp, jm, backend = params

    #### 1. Build potential vorticity anomaly Q = q - f0, zero mean
    @tturbo for j in 1:Ny, i in 1:Nx
        Q[i,j] = q[i,j] - f0
    end
    Q .-= sum(Q) / (Nx*Ny)

    #### 2. Invert Laplacian to get ψ_v
    @views Q_vec .= reshape(Q, :)
    ψ_v_vec .= L_fac \ Q_vec
    @views ψ_v .= reshape(ψ_v_vec, Nx, Ny)

    #### 3. Forcing streamfunction
    ψ_f .= wavemaker(ψ0, x, y, x0, y0, δx, δy, t, τ)

    #### 4. q_f = -L ψ_f
    @views ψ_f_vec .= reshape(ψ_f, :)
    q_f_vec = -(L_op * ψ_f_vec)
    @views q_f .= reshape(q_f_vec, Nx, Ny)

    #### 5. Arakawa Jacobian J(ψ_v + ψ_f, q + q_f)
    # Inputs must be pre-summed
    ψ_sum .= ψ_v .+ ψ_f
    q_sum .= q    .+ q_f    

    k = arakawa_kernel!(params.backend)

    event = k(
        dqdt,
        ψ_sum,      # (already contains ψ_v + ψ_f)
        q_sum,        # (already contains q + q_f)
        params.ip, params.im, params.jp, params.jm,
        params.dx, params.dy;
        ndrange = params.Nx * params.Ny
    )

    KernelAbstractions.synchronize(params.backend)

    #### 6. Linear drag
    @tturbo for j in 1:Ny, i in 1:Nx
        dqdt[i,j] -= r * q[i,j]
    end

    # ---- hyperviscosity (4th order) ----

    # 1. Compute Laplacian(q)
    lap_q_vec .= params.Lap_op * reshape(q, :)

    # 2. Compute Laplacian(Laplacian(q)) = ∇⁴ q
    lap2_q_vec .= params.Lap_op * lap_q_vec

    # 3. Add hyperviscous tendency
    dqdt_vec = reshape(dqdt, :)
    dqdt_vec .-= ν * lap2_q_vec
    dqdt .= reshape(dqdt_vec, Nx, Ny)

    return dqdt
end


