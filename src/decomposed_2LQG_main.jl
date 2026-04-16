

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

function build_consistent_2D_Laplacian(Nx, Ny, dx, dy)
    inv_dx2 = 1.0 / dx^2
    inv_dy2 = 1.0 / dy^2

    # 1. Build 1D x-operator (Periodic)
    Lx = spzeros(Nx, Nx)
    for i in 1:Nx
        im1 = i == 1 ? Nx : i - 1
        ip1 = i == Nx ? 1 : i + 1
        Lx[i, im1] = 1.0 * inv_dx2
        Lx[i, i]   = -2.0 * inv_dx2
        Lx[i, ip1] = 1.0 * inv_dx2
    end

    # 2. Build 1D y-operator (Dirichlet/Consistent Sink)
    Ly = spzeros(Ny, Ny)
    for j in 2:Ny-1
        Ly[j, j-1] = 1.0 * inv_dy2
        Ly[j, j]   = -2.0 * inv_dy2
        Ly[j, j+1] = 1.0 * inv_dy2
    end
    Ly[1, 1] = -2.0 * inv_dy2; Ly[1, 2] = 1.0 * inv_dy2
    Ly[Ny, Ny] = -2.0 * inv_dy2; Ly[Ny, Ny-1] = 1.0 * inv_dy2

    # 3. Combine using Kronecker
    Iy = sparse(I, Ny, Ny)
    Ix = sparse(I, Nx, Nx)
    L2D = kron(Iy, Lx) + kron(Ly, Ix)

    # --- The Dimension-Aware Wrapper ---
    function Lap(ψ_2D)
        # 1. Flatten the 2D input to a 1D vector
        ψ_vec = vec(ψ_2D)
        
        # 2. Multiply by the sparse matrix
        res_vec = L2D * ψ_vec
        
        # 3. Reshape back to the original Nx x Ny dimensions
        return reshape(res_vec, Nx, Ny)
    end
    
    return Lap
end

function build_Laplacian_Neumann(ny, dy)
    inv_dy2 = 1.0 / dy^2
    L_mat = zeros(ny, ny)
    
    # 1. Interior (Standard)
    for j = 2:ny-1
        L_mat[j, j-1] = inv_dy2
        L_mat[j, j]   = -2.0 * inv_dy2
        L_mat[j, j+1] = inv_dy2
    end

    # 2. Boundary: Ghost point reflection (∂y ψ = 0)
    # This represents a "Free-Slip" condition where U = 0 at the wall 
    # but vorticity can exist.
    L_mat[1, 1] = -2.0 * inv_dy2
    L_mat[1, 2] =  2.0 * inv_dy2
    
    L_mat[ny, ny]   = -2.0 * inv_dy2
    L_mat[ny, ny-1] =  2.0 * inv_dy2

    return ψ -> L_mat * ψ
end



# Helper function to handle 1D and 2D arrays
function zero_boundaries!(arr)
    if ndims(arr) == 2
        # For 2D Matrix (q_prime)
        arr[:, 1] .= 0.0
        arr[:, end] .= 0.0
    else
        # For 1D Vector (q_bar)
        arr[1] = 0.0
        arr[end] = 0.0
    end
end


function hyperviscous(q; L = L2D)
    # 1. First Laplacian ∇²q
    # If L is the "Consistent" matrix, it already handles the "Mop"
    lap1 = L(q)
    
    # 2. Second Laplacian ∇⁴q
    # This enforces the Feldstein & Held intermediate zeroing
    if ndims(lap1) == 2
        # lap1[:, 1] .= 0.0
        # lap1[:, end] .= 0.0

        lap1[:,1] = lap1[:,2]
        lap1[:,end] = lap1[:,end-1]
    else
        # lap1[1] = 0.0
        # lap1[end] = 0.0

        lap1[1] = lap1[2]
        lap1[end] = lap1[end-1]
    end
    lap2 = L(lap1)
    
    # 3. Third Laplacian ∇⁶q
    # Again, zero the boundaries to satisfy the 3rd-derivative BC
    if ndims(lap2) == 2
        # lap2[:, 1] .= 0.0
        # lap2[:, end] .= 0.0

        lap2[:,1] = lap2[:,2]
        lap2[:,end] = lap2[:,end-1]
    else
        # lap2[1] = 0.0
        # lap2[end] = 0.0

        lap2[1] = lap2[2]
        lap2[end] = lap2[end-1]
    end
    
    # Apply coefficient and return
    return L(lap2)
end

################################################################################
# define operators used throughout
################################################################################

# function for Laplacian function, L2D(ψ)
# L2D = laplacian_operator_free_slip_y(Nx, Ny, dx, dy)
L2D = build_consistent_2D_Laplacian(Nx, Ny, dx, dy)

L1D = build_Laplacian_Neumann(Ny, dy) # build_Laplacian_with_lower_bc(Ny, dy)


################################################################################
#  Arakawa Jacobian
################################################################################

function arakawa_jacobian(a, b, dx, dy)
    Nx, Ny = size(a)
    J = zeros(Float64, Nx, Ny)
    denom = 4dx * dy

    ip = [i == Nx ? 1 : i + 1 for i in 1:Nx]
    im = [i == 1  ? Nx : i - 1 for i in 1:Nx]

    @threads for i in 1:Nx
        # ONLY loop from 2 to Ny-1. 
        # The wall values (1 and Ny) remain 0.0.
        @inbounds for j in 2:Ny-1
            j_p, j_m = j + 1, j - 1
            i_p, i_m = ip[i], im[i]

            # J1: Standard
            j1 = ((a[i_p, j] - a[i_m, j]) / 2dx) * ((b[i, j_p] - b[i, j_m]) / 2dy) -
                 ((a[i, j_p] - a[i, j_m]) / 2dy) * ((b[i_p, j] - b[i_m, j]) / 2dx)

            # J2 & J3: Flux forms
            j2 = (a[i_p, j] * (b[i_p, j_p] - b[i_p, j_m]) - 
                  a[i_m, j] * (b[i_m, j_p] - b[i_m, j_m]) - 
                  a[i, j_p] * (b[i_p, j_p] - b[i_m, j_p]) + 
                  a[i, j_m] * (b[i_p, j_m] - b[i_m, j_m])) / denom

            j3 = (a[i_p, j_p] * b[i, j_p] - a[i_m, j_p] * b[i, j_p] - 
                  a[i_p, j_m] * b[i, j_m] + a[i_m, j_m] * b[i, j_m] - 
                  a[i_p, j_p] * b[i_p, j] + a[i_p, j_m] * b[i_p, j] + 
                  a[i_m, j_p] * b[i_m, j] - a[i_m, j_m] * b[i_m, j]) / denom

            J[i, j] = (j1 + j2 + j3) / 3.0
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

    q1 = L1D(ψ1) .+ F1 .* (ψ2 .- ψ1) ./ 2
    q2 = L1D(ψ2) .+ F2 .* (ψ1 .- ψ2) ./ 2

    return q1, q2
end


function compute_qg_pv_prime(ψ1::Array{Float64,2}, ψ2::Array{Float64,2}; lap_op=L2D)

    Nx, Ny = size(ψ1)

    q1 = zeros(Nx, Ny)
    q2 = zeros(Nx, Ny)

    # Compute PV; note that we don't add beta, as this makes PV inversion inconsistent
    q1 = lap_op(ψ1) .+ F1 .* (ψ2 .- ψ1) ./ 2 # 
    q2 = lap_op(ψ2) .+ F2 .* (ψ1 .- ψ2) ./ 2  # 

    return q1, q2
end

function precompute_qg_operator(Nx::Int, Ny::Int, F1::Float64, F2::Float64, dx::Float64, dy::Float64)
    N = Nx * Ny
    idx(i, j) = (j - 1) * Nx + i

    inv_dx2 = 1.0 / dx^2
    inv_dy2 = 1.0 / dy^2
    center  = -2 * inv_dx2 - 2 * inv_dy2 

    # We build the 4 blocks of the 2-layer system directly
    A11 = spzeros(N, N)
    A12 = spzeros(N, N)
    A21 = spzeros(N, N)
    A22 = spzeros(N, N)

    for j in 1:Ny
        for i in 1:Nx
            n = idx(i, j)
            
            if j == 1 || j == Ny
                # --- BOUNDARY ROWS ---
                # Force ψ1 = 0 and ψ2 = 0 at the walls.
                # We set the diagonal to 1 and leave coupling blocks as 0.
                A11[n, n] = 1.0
                A22[n, n] = 1.0
                # A12 and A21 rows remain zero here, pinning the layers
            else
                # --- INTERIOR ROWS ---
                i_w = (i == 1)  ? Nx : i-1
                i_e = (i == Nx) ? 1  : i+1
                
                # Layer 1 block (A11)
                A11[n, n]           = center - (F1 / 2.0)
                A11[n, idx(i_w, j)] = inv_dx2
                A11[n, idx(i_e, j)] = inv_dx2
                A11[n, idx(i, j-1)] = inv_dy2
                A11[n, idx(i, j+1)] = inv_dy2
                
                # Layer 1-2 coupling (A12)
                A12[n, n]           = (F1 / 2.0)
                
                # Layer 2-1 coupling (A21)
                A21[n, n]           = (F2 / 2.0)
                
                # Layer 2 block (A22)
                A22[n, n]           = center - (F2 / 2.0)
                A22[n, idx(i_w, j)] = inv_dx2
                A22[n, idx(i_e, j)] = inv_dx2
                A22[n, idx(i, j-1)] = inv_dy2
                A22[n, idx(i, j+1)] = inv_dy2
            end
        end
    end

    # Concatenate into the full system matrix
    A = [A11 A12; A21 A22]

    # This will now be non-singular and well-conditioned
    A_lu = lu(A)

    # Preallocate vectors for performance
    rhs = zeros(2 * N)
    ψ_vec = zeros(2 * N)

    return A_lu, rhs, ψ_vec
end

A_lu, rhs_pa, ψ_vec = precompute_qg_operator(Nx, Ny, F1, F2, dx, dy)


"""
Solve QG PV inversion using preallocated vectors
"""
function invert_qg_pv_prime(q1p::Matrix{Float64}, q2p::Matrix{Float64},
                             A_lu, rhs::Vector{Float64}, ψ_vec::Vector{Float64})
    N = length(q1p)
    Nx, Ny = size(q1p)

    idx = (i,j) -> (j-1)*Nx + i

    @turbo for i in 1:N
        rhs[i] = q1p[i]
        rhs[N+i] = q2p[i]
    end

    # ensures psi=0 at walls
    for i in 1:Nx
        # South Wall (j=1)
        rhs[i] = 0.0           # Layer 1
        rhs[N + i] = 0.0       # Layer 2
        
        # North Wall (j=Ny)
        idx_north = (Ny-1)*Nx + i
        rhs[idx_north] = 0.0   # Layer 1
        rhs[N + idx_north] = 0.0 # Layer 2
    end

    ψ_vec .= A_lu \ rhs
    
    ψ1p = reshape(ψ_vec[1:N], Nx, Ny)
    ψ2p = reshape(ψ_vec[N+1:end], Nx, Ny)

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
    inv_dy2 = 1.0 / dy^2
    
    # 1. Build a clean 1D Laplacian with Ghost-Point Neumann BCs
    # This enforces ∂y ψ = 0, which means U = 0 at the walls.
    L = zeros(Ny, Ny)
    for j in 2:Ny-1
        L[j, j-1] = inv_dy2
        L[j, j]   = -2.0 * inv_dy2
        L[j, j+1] = inv_dy2
    end

    # Southern Boundary: ψ0 = ψ2 (Ghost point)
    L[1, 1] = -2.0 * inv_dy2
    L[1, 2] =  2.0 * inv_dy2

    # Northern Boundary: ψ_{Ny+1} = ψ_{Ny-1} (Ghost point)
    L[Ny, Ny]   = -2.0 * inv_dy2
    L[Ny, Ny-1] =  2.0 * inv_dy2

    I_N = Matrix(I, Ny, Ny)

    # 2. Build the 2-layer block operator
    A = [ (L - (F1/2)*I_N)    ((F1/2)*I_N) ;
          ((F2/2)*I_N)        (L - (F2/2)*I_N) ]

    # 3. CRITICAL: Regularization
    # Neumann systems are singular (defined up to a constant).
    # We pin the South Wall of Layer 1 to EXACTLY 0 to make it invertible.
    A[1, :] .= 0.0
    A[1, 1]  = 1.0

    return PVBarSolver2L(L, F1, F2, lu(A), Ny)
end

solver2L = PVBarSolver2L(Ny, dy, F1, F2)

function invert_qg_pv_bar2L(solver::PVBarSolver2L, q1_bar, q2_bar)
    Ny = solver.Ny
    rhs = vcat(q1_bar, q2_bar)
    
    # Match the "pinned" row in the matrix (South Wall Layer 1 = 0)
    rhs[1] = 0.0 

    ψ_vec = solver.luA \ rhs

    ψ1_bar = ψ_vec[1:Ny]
    ψ2_bar = ψ_vec[Ny+1:end]

    # Post-process: Since Layer 2 wasn't pinned, it might have a DC offset.
    # Subtract the South Wall value of Layer 2 from itself to keep both at 0.
    ψ2_bar .-= ψ2_bar[1]

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

function rhs_prime(q1_prime, q2_prime, q1_bar, q2_bar, ψ_diff_bg)

    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar)

    filter_qprime!(q1_prime)
    filter_qprime!(q2_prime)

    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    # 1. Full Advection
    J1_tot = arakawa_jacobian(ψ1_bar' .+ ψ1_prime, q1_bar' .+ q1_prime, dx, dy)
    J2_tot = arakawa_jacobian(ψ2_bar' .+ ψ2_prime, q2_bar' .+ q2_prime, dx, dy)
    
    # 2. Subtract zonal mean of the Jacobian to keep it a "perturbation"
    # This prevents the streaks!
    dq1dt = -(J1_tot .- mean(J1_tot, dims=1)) .- beta .* u_from_psi(ψ1_prime)[2]
    dq2dt = -(J2_tot .- mean(J2_tot, dims=1)) .- beta .* u_from_psi(ψ2_prime)[2]

    # 3. Damping (Applied to ALL rows including boundaries)
    dq1dt .-= ν .* hyperviscous(ψ1_prime)
    dq2dt .-= ν .* hyperviscous(ψ2_prime)
    dq2dt .-= r .* L2D(ψ2_prime)

    dq1dt .+=  α * F1 .* (ψ1_prime .- ψ2_prime) ./ 2
    dq2dt .+= -α * F2 .* (ψ1_prime .- ψ2_prime) ./ 2

    dq1dt[:, 1] .= 0.0; dq1dt[:, end] .= 0.0
    dq2dt[:, 1] .= 0.0; dq2dt[:, end] .= 0.0

    return dq1dt, dq2dt
end


function rhs_bar(q1_bar, q2_bar, q1_prime, q2_prime, ψ_diff_bg)
    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar)

    filter_qprime!(q1_prime)
    filter_qprime!(q2_prime)

    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    # Eddy forcing: the zonal mean of the perturbation Jacobian
    J1_eddy_mean = mean(arakawa_jacobian(ψ1_prime, q1_prime, dx, dy), dims=1)'
    J2_eddy_mean = mean(arakawa_jacobian(ψ2_prime, q2_prime, dx, dy), dims=1)'

    dq1dt = -J1_eddy_mean
    dq2dt = -J2_eddy_mean

    # Add 1D Damping
    dq1dt .-= ν .* hyperviscous(ψ1_bar; L=L1D)   # twoD=false, L=L1D)
    dq2dt .-= ν .* hyperviscous(ψ2_bar; L=L1D)  # twoD=false, L=L1D)
    dq2dt .-= r .* L1D(ψ2_bar)

    # Thermal relaxation
    dq1dt .+=  α * F1 .* ((ψ1_bar .- ψ2_bar) ./ 2 .- ψ_diff_bg ./ 2)
    dq2dt .+= -α * F2 .* ((ψ1_bar .- ψ2_bar) ./ 2 .- ψ_diff_bg ./ 2)

    # dq1dt[1] = 0.0; dq1dt[end] = 0.0
    # dq2dt[1] = 0.0; dq2dt[end] = 0.0

    return dq1dt, dq2dt
end

###################################################################################
###################################################################################

function u_from_psi(ψ)
    u = -d_dy(ψ, dy)

    ψ_hat = rfft(ψ, 1)                # (Nx/2+1, Ny)

    v_hat = 1im .* KXr .* ψ_hat       # dψ/dx in spectral space
    v = irfft(v_hat, Nx, 1) |> real   # back to x space

    return u, v
end


function pseudomomentum_budget!(q1_bar, q2_bar, q1_prime, q2_prime, v1ζ1, v2ζ2, v1τ, v2τ, q1Jbar, q2Jbar, dy_v_qpsq1, dy_v_qpsq2, q1τ, q2τ, rq2ζ2, γ1_accum, γ2_accum, u1_accum, u2_accum)

    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar)
    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    γ1 = d_dy(reshape(q1_bar, (1, Ny)), dy) .+ beta
    γ2 = d_dy(reshape(q2_bar, (1, Ny)), dy) .+ beta

    γ1_accum .+= γ1[:]
    γ2_accum .+= γ2[:]

    ψ1 = ψ1_bar' .+ ψ1_prime
    ψ2 = ψ2_bar' .+ ψ2_prime

    u1, v1 = u_from_psi(ψ1_prime)
    u2, v2 = u_from_psi(ψ2_prime)

    u1t, v1t = u_from_psi(ψ1)
    u2t, v2t = u_from_psi(ψ2)

    u1_accum .+= mean(u1t, dims=1)[:]
    u2_accum .+= mean(u2t, dims=1)[:]

    ## -v_i zeta_i
    v1ζ1 .+= vec(mean(v1 .* L2D(ψ1_prime), dims=1)) # .* γ1
    v2ζ2 .+= vec(mean(v2 .* L2D(ψ2_prime), dims=1)) # .* γ2

    ## v_i (pis1 - psi2) / 2
    v1τ .+= 0.5 * vec(mean(v1 .* (ψ1_prime .- ψ2_prime), dims=1)) # .* γ1
    v2τ .+= 0.5 * vec(mean(v2 .* (ψ1_prime .- ψ2_prime), dims=1)) # .* γ2

    ## q_i J
    J1_tot = arakawa_jacobian(ψ1, q1_prime, dx, dy) # zeros(Nx, Ny)  # 
    J2_tot = arakawa_jacobian(ψ2, q2_prime, dx, dy) # zeros(Nx, Ny)  # 

    q1Jbar .+= vec(mean(q1_prime .* J1_tot, dims=1)) # ./ γ1)   # this is equivalent to -\partial_{y} \overbar[v (q')^2 / (2 \gamma)]
    q2Jbar .+= vec(mean(q2_prime .* J2_tot, dims=1)) # ./ γ2)

    ## flux form of Jacobian terms (combines them all)
    dy_v_qpsq1 .+= vec(d_dy(mean(v1 .* (q1_prime.^2), dims=1) , dy)) #  ./ (2 .* γ1))
    dy_v_qpsq2 .+= vec(d_dy(mean(v2 .* (q2_prime.^2), dims=1) , dy)) #  ./ (2 .* γ2))

    ## r_T q_i tau / gamma_i
    q1τ .+= vec(mean(v1 .* q1_prime, dims=1))  # vec(mean(α * F1 .* q1_prime .* (ψ1_prime .- ψ2_prime) ./ 2, dims=1)) # ./ γ1)
    q2τ .+= vec(mean(v2 .* q2_prime, dims=1))  #  vec(mean(α * F1 .* q2_prime .* (ψ1_prime .- ψ2_prime) ./ 2, dims=1) ) # ./ γ2)

    ## -r_B q2 zeta2 / gamma_2   [lower layer only]
    rq2ζ2 .+= vec(mean(r .* q2_prime .* L2D(ψ2_prime), dims=1)) # ./ γ2)

    return nothing
end




function energy_budget(q1_bar, q2_bar, q1_prime, q2_prime, save_ind_start, save_ind_end, CBC, CBT, therm_damping, mech_damping)

    ψ1_bar, ψ2_bar = invert_qg_pv_bar2L(solver2L, q1_bar, q2_bar)
    ψ1_prime, ψ2_prime = invert_qg_pv_prime(q1_prime, q2_prime, A_lu, rhs_pa, ψ_vec)

    γ1 = d_dy(reshape(q1_bar, (1, Ny)), dy) .+ beta
    γ2 = d_dy(reshape(q2_bar, (1, Ny)), dy) .+ beta

    ψ1 = ψ1_bar' .+ ψ1_prime
    ψ2 = ψ2_bar' .+ ψ2_prime

    u1_prime, v1_prime = u_from_psi(ψ1_prime)
    u2_prime, v2_prime = u_from_psi(ψ2_prime)

    u1t, v1t = u_from_psi(ψ1)
    u2t, v2t = u_from_psi(ψ2)

    u1_bar = mean(u1t, dims=1)
    u2_bar = mean(u2t, dims=1)

    temp = (u1_bar .- u2_bar) .* mean(ψ1_prime .* v2_prime, dims=1) ./ 2

    CBC += mean(temp[save_ind_start:save_ind_end])

    temp = u1_bar .* d_dy(mean(u1_prime .* v1_prime, dims=1), dy) .+ u2_bar .* d_dy(mean(u2_prime .* v2_prime, dims=1), dy)

    CBT += mean(temp[save_ind_start:save_ind_end])

    temp = α * F1 .* mean(((ψ1_prime .- ψ2_prime).^ 2) ./ 2, dims=1) 

    therm_damping -= mean(temp[save_ind_start:save_ind_end])

    temp = r .* mean(u2_prime.^2 .+ v2_prime.^2, dims=1)

    mech_damping -= mean(temp[save_ind_start:save_ind_end])

    return CBC, CBT, therm_damping, mech_damping
end

