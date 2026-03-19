

################################################################################
# Operations
################################################################################

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
            @turbo for i in 1:Nx
                ip = (i == Nx) ? 1 : i + 1
                im = (i == 1)  ? Nx : i - 1
        
                d2ψ_dx2 = (ψ[ip,j] - 2ψ[i,j] + ψ[im,j]) / dx2
                d2ψ_dy2 = (ψ[i,j+1] - 2ψ[i,j] + ψ[i,j-1]) / dy2
        
                lap[i,j] = d2ψ_dx2 + d2ψ_dy2
            end
        end

        # Southern wall (j=1) free-slip BC
        @turbo for i in 1:Nx
            ip = (i == Nx) ? 1 : i + 1
            im = (i == 1)  ? Nx : i - 1
        
            d2ψ_dx2 = (ψ[ip,1] - 2ψ[i,1] + ψ[im,1]) / dx2
            d2ψ_dy2 = 2*(ψ[i,2] - ψ[i,1]) / dy2
        
            lap[i,1] = d2ψ_dx2 + d2ψ_dy2
        end

        # NOrthern
        @turbo for i in 1:Nx
            ip = (i == Nx) ? 1 : i + 1
            im = (i == 1)  ? Nx : i - 1
        
            d2ψ_dx2 = (ψ[ip,Ny] - 2ψ[i,Ny] + ψ[im,Ny]) / dx2
            d2ψ_dy2 = 2*(ψ[i,Ny-1] - ψ[i,Ny]) / dy2
        
            lap[i,Ny] = d2ψ_dx2 + d2ψ_dy2
        end

        return lap
    end

    return L
end

function build_Laplacian_with_lower_bc(ny, dy)

    function Lap(ψ)
        # Build 1D Laplacian in y with Dirichlet at southern boundary and free at northern
        L = zeros(ny, ny)
        
        # Interior points
        for j = 2:ny-1
            L[j,j]   = -2/dy^2
            L[j,j-1] = 1/dy^2
            L[j,j+1] = 1/dy^2
        end

        # Lower boundary y=0 (free-slip)
        L[1,1] = -2/dy^2
        L[1,2] =  2/dy^2
        
        # Upper boundary y=1 (free, use one-sided derivative approximation for Laplacian)
        L[ny, ny]   = -2/dy^2
        L[ny, ny-1] = 2/dy^2  # one-sided second derivative
        # upper boundary is damped toward psi_bg

        return L * ψ
    end

    return Lap
end


function hyperviscous(q; twoD=true, L=L2D) # q or psi

    lap1 = L(q)
    lap2 = L(lap1)

    return L(lap2)
end

################################################################################
# define operators used throughout
################################################################################

# function for Laplacian function, L2D(ψ)
L2D = laplacian_operator_free_slip_y(Nx, Ny, dx, dy)

L1D = build_Laplacian_with_lower_bc(Ny, dy)


################################################################################
#  Arakawa Jacobian
################################################################################

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

################################################################################
#  2LQG main functions
################################################################################


function compute_qg_pv_bar(ψ1, ψ2; lap_op=L1D)

    Ny = length(ψ1)

    q1 = zeros(Ny)
    q2 = zeros(Ny)

    q1 = L1D(ψ1) .+ F1 .* (ψ2 .- ψ1)
    q2 = L1D(ψ2) .+ F2 .* (ψ1 .- ψ2)

    return q1, q2
end



function compute_qg_pv_prime(ψ1::Array{Float64,2}, ψ2::Array{Float64,2}; lap_op=L2D)

    Nx, Ny = size(ψ1)

    q1 = zeros(Nx, Ny)
    q2 = zeros(Nx, Ny)

    # Compute PV; note that we don't add beta, as this makes PV inversion inconsistent
    q1 = lap_op(ψ1) .+ F1 .* (ψ2 .- ψ1) # 
    q2 = lap_op(ψ2) .+ F2 .* (ψ1 .- ψ2)  # 

    return q1, q2
end


"""
Precompute two-layer QG PV inversion operator with preallocated structures.
Returns:
- LU factorization for repeated solves
- preallocated vectors for rhs and solution
"""
function precompute_qg_operator(Nx::Int, Ny::Int, F1::Float64, F2::Float64, dx::Float64, dy::Float64)
    N = Nx*Ny
    idx = (i,j) -> (j-1)*Nx + i

    dx2, dy2 = dx^2, dy^2
    inv_dx2, inv_dy2 = 1/dx2, 1/dy2
    center = -2*inv_dx2 - 2*inv_dy2 

    L = spzeros(N,N)

    # Threaded assembly
   for j in 1:Ny
        for i in 1:Nx
            n = idx(i,j)
    
            i_w = (i == 1)  ? Nx : i-1
            i_e = (i == Nx) ? 1  : i+1
    
            if j == 1
                # southern free-slip
                L[n,n]           = center # -2*inv_dx2 - inv_dy2 # 
                L[n, idx(i_w,j)] += inv_dx2
                L[n, idx(i_e,j)] += inv_dx2
                L[n, idx(i,j+1)] += 2*inv_dy2
            elseif j == Ny
                # northern free-slip
                L[n,n]           = center # -2*inv_dx2 - inv_dy2 # 
                L[n, idx(i_w,j)] += inv_dx2
                L[n, idx(i_e,j)] += inv_dx2
                L[n, idx(i,j-1)] += 2*inv_dy2
            else
                # interior
                L[n,n]           = center
                L[n, idx(i_w,j)] += inv_dx2
                L[n, idx(i_e,j)] += inv_dx2
                L[n, idx(i,j-1)] += inv_dy2
                L[n, idx(i,j+1)] += inv_dy2
            end
        end
    end

    I_N = spdiagm(0 => ones(N))
    A11 = L - F1*I_N
    A12 = F1*I_N
    A21 = F2*I_N
    A22 = L - F2*I_N

    A = [A11 A12; A21 A22]

    # LU factorization for repeated solves
    A_lu = lu(A)

    # preallocate rhs and solution vectors
    rhs = zeros(2*N)
    ψ_vec = zeros(2*N)

    return A_lu, rhs, ψ_vec
end

A_lu, rhs_pa, ψ_vec = precompute_qg_operator(Nx, Ny, F1, F2, dx, dy)


"""
Solve QG PV inversion using preallocated vectors
"""
function invert_qg_pv_prime(q1p::Matrix{Float64}, q2p::Matrix{Float64},
                             A_lu, rhs::Vector{Float64}, ψ_vec::Vector{Float64})
    N = length(q1p)

    @turbo for i in 1:N
        rhs[i] = q1p[i]
        rhs[N+i] = q2p[i]
    end

    ψ_vec .= A_lu \ rhs

    Nx, Ny = size(q1p)
    ψ1p = reshape(ψ_vec[1:N], Nx, Ny)
    ψ2p = reshape(ψ_vec[N+1:end], Nx, Ny)

    ψ1p = ψ1p .- mean(ψ1p, dims=1)
    ψ2p = ψ2p .- mean(ψ2p, dims=1)

    return ψ1p, ψ2p
end


struct PVBarSolver2L
    L::Matrix{Float64}       # 1D Laplacian
    F1::Float64
    F2::Float64
    luA::LU                   # LU factorization of 2-layer operator
    Ny::Int
end

function PVBarSolver2L(Ny::Int, dy::Float64, F1::Float64, F2::Float64)
    # 1D Laplacian in y
    main_diag = -2.0 * ones(Ny) / dy^2
    off_diag  = ones(Ny-1) / dy^2
    L = Matrix(Tridiagonal(off_diag, main_diag, off_diag))

    # FIX: free slip instead of above code
    L[1,1] = -2/dy^2
    L[1,2] =  2/dy^2

    # Second-order Neumann top: (-psi_{N-2} + 4*psi_{N-1} - 3*psi_N)/(2*dy) = 0
    L[Ny, Ny]   = -3.0 / (2*dy)
    L[Ny, Ny-1] = 4.0 / (2*dy)
    L[Ny, Ny-2] = -1.0 / (2*dy)

    I_N = Matrix(I, Ny, Ny)

    # 2-layer block operator
    L1 = L - F1*I_N
    L2 = L - F2*I_N
    C12 = F1 * I_N
    C21 = F2 * I_N

    A = [L1  C12;
         C21 L2]

    # LU factorization
    luA = lu(A)

    return PVBarSolver2L(L, F1, F2, luA, Ny)
end

solver2L = PVBarSolver2L(Ny, dy, F1, F2)

"""
Invert two-layer zonal-mean PV to get streamfunction for each layer.
- q1_bar, q2_bar: Ny-element vectors of zonal-mean PV
- ψ_bg_lower: Dirichlet value at bottom of layer 1
"""
function invert_qg_pv_bar2L(solver::PVBarSolver2L, q1_bar, q2_bar, ψ_bg_lower::Float64)
    Ny = solver.Ny

    # Build RHS vector
    q_vec = vcat(q1_bar, q2_bar)

    # Solve
    ψ_vec = solver.luA \ q_vec

    # Extract layers
    ψ1_bar = ψ_vec[1:Ny]
    ψ2_bar = ψ_vec[Ny+1:end]

    # FIX: remove barotropic drift
    ψ1_bar .-= mean(ψ1_bar)
    ψ2_bar .-= mean(ψ2_bar)

    return ψ1_bar, ψ2_bar
end


function filter_qprime!(q)
    # Remove zonal mean; shouldn't have one anyways
    q .-= mean(q, dims=1)

    # Kill x-Nyquist mode (checkerboard)
    q̂ = rfft(q, 1)
    q̂[end, :] .= 0.0
    q .= irfft(q̂, size(q,1), 1)
end



#####################################################################
function rhs_prime(q1_prime, q2_prime, q1_bar, q2_bar)
    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar, ψ1_bg[1])

    # removes checkerboard mode
    filter_qprime!(q1_prime)
    filter_qprime!(q2_prime)

    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)
    # ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, inversion_ops, rhs_pa, ψ_vec)

    # ψ1_prime, ψ2_prime = invert_qg_pv(q1_prime, q2_prime, ψ1_bar, ψ2_bar, inversion_ops, dx, dy) 

    J1_bar = mean(arakawa_jacobian(ψ1_prime, q1_prime, dx, dy), dims=1) # zeros(Nx, Ny)  # 
    J2_bar = mean(arakawa_jacobian(ψ2_prime, q2_prime, dx, dy), dims=1) # zeros(Nx, Ny)  # 

    J1_tot = arakawa_jacobian(ψ1_bar' .+ ψ1_prime, q1_bar' .+ q1_prime, dx, dy) # zeros(Nx, Ny)  # 
    J2_tot = arakawa_jacobian(ψ2_bar' .+ ψ2_prime, q2_bar' .+ q2_prime, dx, dy) # zeros(Nx, Ny)  # 

    dq1dt = J1_bar .- J1_tot .- beta .* u_from_psi(ψ1_prime)[2]
    dq2dt = J2_bar .- J2_tot .- beta .* u_from_psi(ψ2_prime)[2]

    dq1dt .-= ν .* hyperviscous(ψ1_prime)
    dq2dt .-= ν .* hyperviscous(ψ2_prime)
    
    dq2dt .-= r .* L2D(ψ2_prime)

    # # Thermal damping toward background shear; perturbation get damped to zero
    dq1dt .+=  α * F1 .* (ψ1_prime .- ψ2_prime)
    dq2dt .+= -α * F2 .* (ψ1_prime .- ψ2_prime)

    return dq1dt, dq2dt
end

function rhs_bar(q1_bar, q2_bar, q1_prime, q2_prime)
    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar, ψ1_bg[1])

    filter_qprime!(q1_prime)
    filter_qprime!(q2_prime)

    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    J1_bar = mean(arakawa_jacobian(ψ1_prime, q1_prime, dx, dy), dims=1)' # zeros(Nx, Ny)  # 
    J2_bar = mean(arakawa_jacobian(ψ2_prime, q2_prime, dx, dy), dims=1)' # zeros(Nx, Ny)  # 

    dq1dt = -J1_bar
    dq2dt = -J2_bar

    dq1dt .-= ν .* hyperviscous(ψ1_bar; twoD=false, L=L1D)
    dq2dt .-= ν .* hyperviscous(ψ2_bar; twoD=false, L=L1D)
    
    dq2dt .-= r .* L1D(ψ2_bar)

    # Thermal damping toward background shear
    dq1dt .+=  α * F1 .* ((ψ1_bar .- ψ2_bar) .- ψ_diff_bg)
    dq2dt .+= -α * F2 .* ((ψ1_bar .- ψ2_bar) .- ψ_diff_bg)

    return dq1dt, dq2dt
end


function u_from_psi(ψ)
    u = -d_dy(ψ, dy)

    ψ_hat = rfft(ψ, 1)                # (Nx/2+1, Ny)

    v_hat = 1im .* KXr .* ψ_hat       # dψ/dx in spectral space
    v = irfft(v_hat, Nx, 1) |> real   # back to x space

    return u, v
end


# function pseudomomentum_budget!(q1_bar, q2_bar, q1_prime, q2_prime, v1ζ1, v2ζ2, v1τ, v2τ, q1Jbar, q2Jbar, q1τ, q2τ, rq2ζ2)

#     ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar, ψ1_bg[1])
#     ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

#     γ1 = d_dy(reshape(q1_bar, (1, Ny)), dy) .+ beta
#     γ2 = d_dy(reshape(q2_bar, (1, Ny)), dy) .+ beta

#     ψ1 = ψ1_bar' .+ ψ1_prime
#     ψ2 = ψ2_bar' .+ ψ2_prime

#     q1 = q1_bar' .+ q1_prime
#     q2 = q2_bar' .+ q2_prime

#     u1, v1 = u_from_psi(ψ1)
#     u2, v2 = u_from_psi(ψ2)

#     ## -v_i zeta_i

#     v1ζ1 .= vec(mean(v1 .* L2D(ψ1), dims=1))
#     v2ζ2 .= vec(mean(v2 .* L2D(ψ2), dims=1))

#     ## v_i (pis1 - psi2) / 2

#     v1τ .= 0.5 * vec(mean(v1 .* (ψ1 .- ψ2), dims=1))
#     v2τ .= 0.5 * vec(mean(v2 .* (ψ1 .- ψ2), dims=1))

#     ## q_i J ( psi_i, q_i)  [this is full psi_i]

#     # J1_bar = mean(arakawa_jacobian(ψ1_prime, q1_prime, dx, dy), dims=1) # zeros(Nx, Ny)  # 
#     # J2_bar = mean(arakawa_jacobian(ψ2_prime, q2_prime, dx, dy), dims=1) # zeros(Nx, Ny)  # 

#     J1_tot = arakawa_jacobian(ψ1, q1, dx, dy) # zeros(Nx, Ny)  # 
#     J2_tot = arakawa_jacobian(ψ2, q2, dx, dy) # zeros(Nx, Ny)  # 

#     q1Jbar .= vec(mean(q1 .* J1_tot, dims=1) ./ γ1)
#     q2Jbar .= vec(mean(q2 .* J2_tot, dims=1) ./ γ2)

#     ## r_T q_i tau / gamma_i
#     q1τ .=  vec(mean(α * F1 .* q1 .* (ψ1 .- ψ2), dims=1) ./ γ1)
#     q2τ .= vec(mean(α * F1 .* q2 .* (ψ1 .- ψ2), dims=1) ./ γ2)

#     ## -r_B q2 zeta2 / gamma_2   [lower layer only]
#     rq2ζ2 .= vec(mean(r .* q2 .* L2D(ψ2), dims=1) ./ γ2)

#     return nothing
# end


function pseudomomentum_budget!(q1_bar, q2_bar, q1_prime, q2_prime, v1ζ1, v2ζ2, v1τ, v2τ, q1Jbar, q2Jbar, dy_v_qpsq1, dy_v_qpsq2, q1τ, q2τ, rq2ζ2)

    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar, ψ1_bg[1])
    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    γ1 = d_dy(reshape(q1_bar, (1, Ny)), dy) .+ beta
    γ2 = d_dy(reshape(q2_bar, (1, Ny)), dy) .+ beta

    ψ1 = ψ1_bar' .+ ψ1_prime
    ψ2 = ψ2_bar' .+ ψ2_prime

    u1, v1 = u_from_psi(ψ1_prime)
    u2, v2 = u_from_psi(ψ2_prime)

    ## -v_i zeta_i
    v1ζ1 .= vec(mean(v1 .* L2D(ψ1_prime), dims=1))
    v2ζ2 .= vec(mean(v2 .* L2D(ψ2_prime), dims=1))

    ## v_i (pis1 - psi2) / 2
    v1τ .= 0.5 * vec(mean(v1 .* (ψ1_prime .- ψ2_prime), dims=1))
    v2τ .= 0.5 * vec(mean(v2 .* (ψ1_prime .- ψ2_prime), dims=1))

    ## q_i J
    J1_tot = arakawa_jacobian(ψ1, q1_prime, dx, dy) # zeros(Nx, Ny)  # 
    J2_tot = arakawa_jacobian(ψ2, q2_prime, dx, dy) # zeros(Nx, Ny)  # 

    q1Jbar .= vec(mean(q1_prime .* J1_tot, dims=1) ./ γ1)   # this is equivalent to -\partial_{y} \overbar[v (q')^2 / (2 \gamma)]
    q2Jbar .= vec(mean(q2_prime .* J2_tot, dims=1) ./ γ2)

    ## flux form of Jacobian terms (combines them all)
    dy_v_qpsq1 .= vec(mean(q1_prime .* J1_tot, dims=1))  # vec(d_dy(mean(v1 .* (q1_prime.^2), dims=1) , dy)  ./ (2 .* γ1))
    dy_v_qpsq2 .= vec(mean(q2_prime .* J2_tot, dims=1))  # vec(d_dy(mean(v2 .* (q2_prime.^2), dims=1) , dy)  ./ (2 .* γ2))

    ## r_T q_i tau / gamma_i
    q1τ .=  vec(mean(α * F1 .* q1_prime .* (ψ1_prime .- ψ2_prime), dims=1) ./ γ1)
    q2τ .= vec(mean(α * F1 .* q2_prime .* (ψ1_prime .- ψ2_prime), dims=1) ./ γ2)

    ## -r_B q2 zeta2 / gamma_2   [lower layer only]
    rq2ζ2 .= vec(mean(r .* q2_prime .* L2D(ψ2_prime), dims=1) ./ γ2)

    return nothing
end


