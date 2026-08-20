#######################################################################
## Helpful general structs
#######################################################################

# 1. Define PVBarSolver2L FIRST so QGOperators knows what it is!
struct PVBarSolver2L
    L::Matrix{Float64}        # 1D Laplacian
    F1::Float64
    F2::Float64
    luA::LU                   # LU factorization of 2-layer operator
    Ny::Int
end

# 2. The Evolving State
mutable struct QGState
    # Prognostic PV
    q1_prime::Matrix{Float64}
    q2_prime::Matrix{Float64}
    q1_bar::Vector{Float64}
    q2_bar::Vector{Float64}
    
    # Diagnostic Streamfunctions
    ψ1_prime::Matrix{Float64}
    ψ2_prime::Matrix{Float64}
    ψ1_bar::Vector{Float64}
    ψ2_bar::Vector{Float64}
end

# 3. The Static Operators and Buffers
# We use type parameters {TLU, T1D, T2D} because functions and LU 
# factorizations have complex types. This guarantees type-stability.
struct QGOperators{TLU, T1D, T2D}
    solver2L::PVBarSolver2L
    A_lu::TLU
    rhs_pa::Vector{Float64}  # Pre-allocated buffer for RHS
    ψ_vec::Vector{Float64}   # Pre-allocated buffer for Psi
    L1D::T1D                 # 1D Laplacian function/matrix
    L2D::T2D                 # 2D Laplacian function/matrix

    # Pre-allocated 2D Jacobian buffers ---
    J1_buffer::Matrix{Float64}
    J2_buffer::Matrix{Float64}
end

# 4. The Parent Wrapper
struct QGModel{TLU, T1D, T2D}
    state::QGState
    ops::QGOperators{TLU, T1D, T2D}
    # You could also add a `grid` or `params` struct here!
end

Base.@kwdef struct OutputConfig
    # Booleans
    save_bool::Bool = true
    save_last::Bool = true
    plot_basic_bool::Bool = true
    plot_BCI_bool::Bool = false
    diag_bool::Bool = true
    nrg_diag_bool::Bool = true

    # Frequencies
    save_every::Int
    plot_every::Int
    diag_every::Int

    # Paths
    save_path::String
    fig_path::String
    diag_dir::String
end

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



function hyperviscous(psi; L = L2D)
    # 1. First Laplacian ∇²psi
    # If L is the "Consistent" matrix, it already handles the "Mop"
    lap1 = L(psi)
    
    # 2. Second Laplacian ∇⁴q
    # This enforces the Feldstein & Held intermediate zeroing
    if ndims(lap1) == 2
        lap1[:,1] .= lap1[:,2]
        lap1[:,end] .= lap1[:,end-1]
    else
        lap1[1] = lap1[2]
        lap1[end] = lap1[end-1]
    end
    lap2 = L(lap1)
    
    # 3. Third Laplacian ∇⁶q
    # Again, zero the boundaries to satisfy the 3rd-derivative BC
    if ndims(lap2) == 2
        lap2[:,1] .= lap2[:,2]
        lap2[:,end] .= lap2[:,end-1]
    else
        lap2[1] = lap2[2]
        lap2[end] = lap2[end-1]
    end
    
    # Apply coefficient and return
    return L(lap2)
end

################################################################################
#  Arakawa Jacobian
################################################################################
function arakawa_jacobian!(J::Matrix{Float64}, a, b, dx, dy)
    Nx, Ny = size(a)
    denom = 4dx * dy

    # Pre-calculate indices to avoid allocating arrays inside the function
    @inbounds for i in 1:Nx
        i_p = i == Nx ? 1 : i + 1
        i_m = i == 1  ? Nx : i - 1
        
        # We only loop the interior y-indices
        for j in 2:Ny-1
            j_p, j_m = j + 1, j - 1

            j1 = ((a[i_p, j] - a[i_m, j]) / 2dx) * ((b[i, j_p] - b[i, j_m]) / 2dy) -
                 ((a[i, j_p] - a[i, j_m]) / 2dy) * ((b[i_p, j] - b[i_m, j]) / 2dx)

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
    q1 = lap_op(ψ1) .+ F1 .* (ψ2 .- ψ1) ./ 2  
    q2 = lap_op(ψ2) .+ F2 .* (ψ1 .- ψ2) ./ 2  

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
                A11[n, n] = 1.0
                A22[n, n] = 1.0
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

    # ensures psi_prime=0 at walls
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
    # q .-= mean(q, dims=1)

    # Kill x-Nyquist mode (checkerboard)
    q̂ = rfft(q, 1)
    q̂[end, :] .= 0.0
    q .= irfft(q̂, size(q,1), 1)
end


#####################################################################
# RHS Tendencies
#####################################################################
#=
function rhs_prime(model::QGModel, ψ_diff_bg, topo_PV)
    # Unpack for cleaner syntax
    s = model.state
    o = model.ops

    # Invert PV (updates the streamfunctions inside the state struct)
    s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)

    filter_qprime!(s.q1_prime)
    filter_qprime!(s.q2_prime)

    s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)


    # 1. DIAGNOSE BOUNDARY PV (Enforces ∂y q' = 0)
    # we enforce this bc q' = zeta' @ wall since psi1= 0 = psi2, and we need ∂y zeta' = 0 @ wall
    s.q1_prime[:, 1] .= s.q1_prime[:, 2]
    s.q1_prime[:, end] .= s.q1_prime[:, end-1]
    s.q2_prime[:, 1] .= s.q2_prime[:, 2]
    s.q2_prime[:, end] .= s.q2_prime[:, end-1]


    # 1. Full Advection
    arakawa_jacobian!(o.J1_buffer, s.ψ1_bar' .+ s.ψ1_prime, s.q1_bar' .+ s.q1_prime, dx, dy)
    arakawa_jacobian!(o.J2_buffer, s.ψ2_bar' .+ s.ψ2_prime, s.q2_bar' .+ s.q2_prime .+ topo_PV, dx, dy)
    
    # 2. Subtract zonal mean using the buffers
    dq1dt = -(o.J1_buffer .- mean(o.J1_buffer, dims=1)) .- beta .* u_from_psi(s.ψ1_prime)[2]
    dq2dt = -(o.J2_buffer .- mean(o.J2_buffer, dims=1)) .- beta .* u_from_psi(s.ψ2_prime)[2]

    # 3. Damping
    dq1dt .-= ν .* hyperviscous(s.ψ1_prime; L=o.L2D)
    dq2dt .-= ν .* hyperviscous(s.ψ2_prime; L=o.L2D)
    dq2dt .-= r .* o.L2D(s.ψ2_prime)

    # Decompose the background field to get the 2D perturbation
    ψ_diff_bg_prime = ψ_diff_bg .- mean(ψ_diff_bg, dims=1)

    # Thermal relaxation (now includes the prime background state)
    dq1dt .+=  α * F1 .* ((s.ψ1_prime .- s.ψ2_prime) ./ 2 .- ψ_diff_bg_prime ./ 2)
    dq2dt .+= -α * F2 .* ((s.ψ1_prime .- s.ψ2_prime) ./ 2 .- ψ_diff_bg_prime ./ 2)

    ## STJ relaxation
    dq1dt .-= α_STJ_1D' .* s.q1_prime
    dq2dt .-= α_STJ_1D' .* s.q2_prime

    dq1dt[:, 1] .= 0.0; dq1dt[:, end] .= 0.0
    dq2dt[:, 1] .= 0.0; dq2dt[:, end] .= 0.0


    return dq1dt, dq2dt
end


function rhs_bar(model::QGModel, ψ_diff_bg, topo_PV, wind_curl, q1_STJ_target, q2_STJ_target)
    # Unpack for cleaner syntax
    s = model.state
    o = model.ops

    # Invert PV (updates the streamfunctions inside the state struct)
    s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)

    filter_qprime!(s.q1_prime)
    filter_qprime!(s.q2_prime)

    s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)

    inv_dy2 = 1.0 / dy^2

    # 1. DIAGNOSE BOUNDARY MEAN PV (Enforces U = 0)
    # bc U = 0 implies d_dy(\overline{psi}) = 2 (\overline{psi1 - psi2})/dy^2 using centered difference and ghost point
    # so that q at the walls is just this term plus the layer-coupling term
    # Layer 1 South & North
    s.q1_bar[1] = 2.0 * inv_dy2 * (s.ψ1_bar[2] - s.ψ1_bar[1]) + 
                  (F1 / 2.0) * (s.ψ2_bar[1] - s.ψ1_bar[1])
    s.q1_bar[end] = 2.0 * inv_dy2 * (s.ψ1_bar[end-1] - s.ψ1_bar[end]) + 
                    (F1 / 2.0) * (s.ψ2_bar[end] - s.ψ1_bar[end])

    # Layer 2 South & North
    s.q2_bar[1] = 2.0 * inv_dy2 * (s.ψ2_bar[2] - s.ψ2_bar[1]) + 
                  (F2 / 2.0) * (s.ψ1_bar[1] - s.ψ2_bar[1])
    s.q2_bar[end] = 2.0 * inv_dy2 * (s.ψ2_bar[end-1] - s.ψ2_bar[end]) + 
                    (F2 / 2.0) * (s.ψ1_bar[end] - s.ψ2_bar[end])

    # Nonlinear advection
    arakawa_jacobian!(o.J1_buffer, s.ψ1_prime, s.q1_prime, dx, dy)
    arakawa_jacobian!(o.J2_buffer, s.ψ2_prime, s.q2_prime .+ topo_PV, dx, dy)

    # Calculate the zonal mean of the filled buffers
    J1_eddy_mean = mean(o.J1_buffer, dims=1)'
    J2_eddy_mean = mean(o.J2_buffer, dims=1)'

    dq1dt = -J1_eddy_mean
    dq2dt = -J2_eddy_mean

    # Add 1D Damping
    dq1dt .-= ν .* hyperviscous(s.ψ1_bar; L=o.L1D)  
    dq2dt .-= ν .* hyperviscous(s.ψ2_bar; L=o.L1D) 
    dq2dt .-= r .* o.L1D(s.ψ2_bar)

    # Decompose the background field to get the 1D zonal mean
    # vec() converts the 1xNy mean into an Ny-element Vector to match ψ1_bar
    ψ_diff_bg_bar = vec(mean(ψ_diff_bg, dims=1))

    # Thermal relaxation (uses the 1D bar background state)
    dq1dt .+=  α * F1 .* ((s.ψ1_bar .- s.ψ2_bar) ./ 2 .- ψ_diff_bg_bar ./ 2)
    dq2dt .+= -α * F2 .* ((s.ψ1_bar .- s.ψ2_bar) ./ 2 .- ψ_diff_bg_bar ./ 2)

    # Surface Wind Forcing ---
    dq1dt .+= wind_curl

    # STJ momentum forcing
    dq1dt .-= α_STJ_1D .* (s.q1_bar .- q1_STJ_target)
    dq2dt .-= α_STJ_1D .* (s.q2_bar .- q2_STJ_target)

    dq1dt[1] = 0.0; dq1dt[end] = 0.0
    dq2dt[1] = 0.0; dq2dt[end] = 0.0


    return dq1dt, dq2dt
end
=#

function rhs_prime(model::QGModel, ψ_diff_bg, topo_PV)
    s = model.state
    o = model.ops

    # Invert PV
    s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)
    s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)

    # Full Advection
    arakawa_jacobian!(o.J1_buffer, s.ψ1_bar' .+ s.ψ1_prime, s.q1_bar' .+ s.q1_prime, dx, dy)
    arakawa_jacobian!(o.J2_buffer, s.ψ2_bar' .+ s.ψ2_prime, s.q2_bar' .+ s.q2_prime .+ topo_PV, dx, dy)

    # Subtract zonal mean using the buffers
    # (The interior of J_buffer is safe; the garbage at boundaries is overwritten below)
    dq1dt = -(o.J1_buffer .- mean(o.J1_buffer, dims=1)) .- beta .* u_from_psi(s.ψ1_prime)[2]
    dq2dt = -(o.J2_buffer .- mean(o.J2_buffer, dims=1)) .- beta .* u_from_psi(s.ψ2_prime)[2]

    # Damping
    dq1dt .-= ν .* hyperviscous(s.ψ1_prime; L=o.L2D)
    dq2dt .-= ν .* hyperviscous(s.ψ2_prime; L=o.L2D)
    dq2dt .-= r .* o.L2D(s.ψ2_prime)

    # Thermal relaxation
    ψ_diff_bg_prime = ψ_diff_bg .- mean(ψ_diff_bg, dims=1)
    dq1dt .+=  α * F1 .* ((s.ψ1_prime .- s.ψ2_prime) ./ 2 .- ψ_diff_bg_prime ./ 2)
    dq2dt .+= -α * F2 .* ((s.ψ1_prime .- s.ψ2_prime) ./ 2 .- ψ_diff_bg_prime ./ 2)

    # STJ relaxation
    dq1dt .-= α_STJ_1D' .* s.q1_prime
    dq2dt .-= α_STJ_1D' .* s.q2_prime

    # ENFORCE NEUMANN BOUNDARIES VIA TENDENCIES (dq/dt |_wall = dq/dt |_interior)
    dq1dt[:, 1] .= dq1dt[:, 2]; dq1dt[:, end] .= dq1dt[:, end-1]
    dq2dt[:, 1] .= dq2dt[:, 2]; dq2dt[:, end] .= dq2dt[:, end-1]

    return dq1dt, dq2dt
end

function rhs_bar(model::QGModel, ψ_diff_bg, topo_PV, wind_curl, q1_STJ_target, q2_STJ_target)
    s = model.state
    o = model.ops

    # Invert PV
    s.ψ1_bar, s.ψ2_bar = invert_qg_pv_bar2L(o.solver2L, s.q1_bar, s.q2_bar)
    s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)

    # Nonlinear advection
    arakawa_jacobian!(o.J1_buffer, s.ψ1_prime, s.q1_prime, dx, dy)
    arakawa_jacobian!(o.J2_buffer, s.ψ2_prime, s.q2_prime .+ topo_PV, dx, dy)

    # Calculate the zonal mean and scrub memory garbage at the walls
    J1_eddy_mean = mean(o.J1_buffer, dims=1)'
    J2_eddy_mean = mean(o.J2_buffer, dims=1)'
    J1_eddy_mean[1] = 0.0; J1_eddy_mean[end] = 0.0
    J2_eddy_mean[1] = 0.0; J2_eddy_mean[end] = 0.0

    dq1dt = -J1_eddy_mean
    dq2dt = -J2_eddy_mean

    # Add 1D Damping
    dq1dt .-= ν .* hyperviscous(s.ψ1_bar; L=o.L1D)
    dq2dt .-= ν .* hyperviscous(s.ψ2_bar; L=o.L1D)
    dq2dt .-= r .* o.L1D(s.ψ2_bar)

    # Thermal relaxation
    ψ_diff_bg_bar = vec(mean(ψ_diff_bg, dims=1))
    dq1dt .+=  α * F1 .* ((s.ψ1_bar .- s.ψ2_bar) ./ 2 .- ψ_diff_bg_bar ./ 2)
    dq2dt .+= -α * F2 .* ((s.ψ1_bar .- s.ψ2_bar) ./ 2 .- ψ_diff_bg_bar ./ 2)

    # Surface Wind Forcing & STJ momentum forcing
    dq1dt .+= wind_curl
    dq1dt .-= α_STJ_1D .* (s.q1_bar .- q1_STJ_target)
    dq2dt .-= α_STJ_1D .* (s.q2_bar .- q2_STJ_target)

    # Let the tendencies at the walls naturally evolve!
    # Removing dq1dt[1] = 0.0 prevents the infinite PV spike.

    return dq1dt, dq2dt
end


###################################################################
# For a fixed background flow (that still allows the zonal mean state to evolve)

function rhs_full(model::QGModel, ψ_bg, q1_bg_grad, q2_bg_grad, topo_PV)
    s = model.state
    o = model.ops

    # 1. Filter the 2D perturbation PV
    filter_qprime!(s.q1_prime)
    filter_qprime!(s.q2_prime)

    # 2. Invert 2D perturbation PV to find perturbation streamfunctions
    s.ψ1_prime, s.ψ2_prime = invert_qg_pv_prime(s.q1_prime, s.q2_prime, o.A_lu, o.rhs_pa, o.ψ_vec)

    Nx, Ny = size(s.q1_prime)
    dq1dt = zeros(Float64, Nx, Ny)
    dq2dt = zeros(Float64, Nx, Ny)

    # 3. Layer 1 Jacobians: J(ψ_bg + ψ', q')
    @. o.J1_buffer = ψ_bg + s.ψ1_prime
    arakawa_jacobian!(dq1dt, o.J1_buffer, s.q1_prime, dx, dy)

    # 4. Layer 2 Jacobians: J(ψ', q' + topo_PV)
    # This perfectly captures non-linear eddy advection AND any x/y topographic gradients!
    @. o.J2_buffer = s.q2_prime + topo_PV        
    arakawa_jacobian!(dq2dt, s.ψ2_prime, o.J2_buffer, dx, dy)

    # 5. Compute velocities for the explicit linear background gradients
    _, v1_prime = u_from_psi(s.ψ1_prime)
    _, v2_prime = u_from_psi(s.ψ2_prime)

    # 6. Apply full background PV gradients (Beta + Jet Curvature + Stretching)
    # Note: NO separate '- beta * v' line needed anymore, beta is inside these gradients!
    @. dq1dt = -dq1dt - q1_bg_grad * v1_prime
    @. dq2dt = -dq2dt - q2_bg_grad * v2_prime

    # 7. Damping
    dq1dt .-= ν .* hyperviscous(s.ψ1_prime; L=o.L2D)
    dq2dt .-= ν .* hyperviscous(s.ψ2_prime; L=o.L2D)
    dq2dt .-= r .* o.L2D(s.ψ2_prime)

    # 8. Boundaries
    dq1dt[:, 1] .= 0.0; dq1dt[:, end] .= 0.0
    dq2dt[:, 1] .= 0.0; dq2dt[:, end] .= 0.0

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

################################################################################
# Initialize Operators (Moved from mid-file to prevent UndefVarErrors)
################################################################################

# NOTE: Since these rely on global variables like Nx, Ny, dx, dy, F1, and F2, 
# they MUST be evaluated *after* those variables are defined in your main script.
L2D = build_consistent_2D_Laplacian(Nx, Ny, dx, dy)
L1D = build_Laplacian_Neumann(Ny, dy) 
solver2L = PVBarSolver2L(Ny, dy, F1, F2)
A_lu, rhs_pa, ψ_vec = precompute_qg_operator(Nx, Ny, F1, F2, dx, dy)




################################################################3333
## Diagnostic suite
##################################################################

abstract type AbstractDiagnostic end

struct DiagnosticSuite
    diags::Dict{Symbol, AbstractDiagnostic}
end

# Constructor for an empty suite
DiagnosticSuite() = DiagnosticSuite(Dict{Symbol, AbstractDiagnostic}())

# Function to add a diagnostic
function add!(suite::DiagnosticSuite, name::Symbol, diag::AbstractDiagnostic)
    suite.diags[name] = diag
end

# Function to run all diagnostics
function compute_all!(suite::DiagnosticSuite, state, t::Float64, step::Int)
    for diag in values(suite.diags)
        compute!(diag, state, t, step)
    end
end


#########################################3
## EKE

mutable struct EKEDiagnostic <: AbstractDiagnostic
    frequency::Int
    counter::Int
    times::Vector{Float64}
    data::Matrix{Float64} # 2 x n_diag
end

# Constructor handles the math for pre-allocation
function EKEDiagnostic(nt::Int, dt::Float64, frequency::Int)
    n_diag = ceil(Int, nt / frequency)
    return EKEDiagnostic(frequency, 1, zeros(n_diag), zeros(2, n_diag))
end

# The dispatch for computing EKE
function compute!(diag::EKEDiagnostic, state::QGState, t::Float64, step::Int)
    if step % diag.frequency == 0
        idx = diag.counter
        
        u1, v1 = u_from_psi(state.ψ1_prime)
        u2, v2 = u_from_psi(state.ψ2_prime)
        
        diag.data[1, idx] = 0.5 * mean(u1.^2 .+ v1.^2)
        diag.data[2, idx] = 0.5 * mean(u2.^2 .+ v2.^2)
        diag.times[idx] = t
        
        diag.counter += 1
    end
end

#########################################3
## Pseudomomentum budget
mutable struct PseudomomentumDiag <: AbstractDiagnostic
    frequency::Int
    counter::Int
    times::Vector{Float64}
    
    # Accumulated 1D arrays
    v1ζ1::Vector{Float64}
    v2ζ2::Vector{Float64}
    v1τ::Vector{Float64}
    v2τ::Vector{Float64}
    q1Jbar::Vector{Float64}
    q2Jbar::Vector{Float64}
    dy_v_qpsq1::Vector{Float64}
    dy_v_qpsq2::Vector{Float64}
    q1τ::Vector{Float64}
    q2τ::Vector{Float64}
    rq2ζ2::Vector{Float64}
    γ1_accum::Vector{Float64}
    γ2_accum::Vector{Float64}
    u1_accum::Vector{Float64}
    u2_accum::Vector{Float64}
end

# Constructor for pre-allocation
function PseudomomentumDiag(nt::Int, dt::Float64, frequency::Int, Ny::Int)
    n_diag = ceil(Int, nt / frequency)
    return PseudomomentumDiag(
        frequency, 1, zeros(n_diag),
        zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny),
        zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny),
        zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny), zeros(Ny)
    )
end

function compute!(diag::PseudomomentumDiag, model::QGModel, t::Float64, step::Int)
    if step % diag.frequency == 0
        s = model.state
        o = model.ops
        
        diag.times[diag.counter] = t
        
        # Background gradients
        γ1 = d_dy(reshape(s.q1_bar, (1, size(s.q1_bar, 1))), dy) .+ beta
        γ2 = d_dy(reshape(s.q2_bar, (1, size(s.q2_bar, 1))), dy) .+ beta
        
        diag.γ1_accum .+= γ1[:]
        diag.γ2_accum .+= γ2[:]
        
        ψ1 = s.ψ1_bar' .+ s.ψ1_prime
        ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        
        u1, v1 = u_from_psi(s.ψ1_prime)
        u2, v2 = u_from_psi(s.ψ2_prime)
        
        u1t, v1t = u_from_psi(ψ1)
        u2t, v2t = u_from_psi(ψ2)
        
        diag.u1_accum .+= mean(u1t, dims=1)[:]
        diag.u2_accum .+= mean(u2t, dims=1)[:]
        
        # Accumulate fluxes and terms
        diag.v1ζ1 .+= vec(mean(v1 .* o.L2D(s.ψ1_prime), dims=1))
        diag.v2ζ2 .+= vec(mean(v2 .* o.L2D(s.ψ2_prime), dims=1))
        
        diag.v1τ .+= 0.5 .* vec(mean(v1 .* (s.ψ1_prime .- s.ψ2_prime), dims=1))
        diag.v2τ .+= 0.5 .* vec(mean(v2 .* (s.ψ1_prime .- s.ψ2_prime), dims=1))
        
        J1_tot = arakawa_jacobian(ψ1, s.q1_prime, dx, dy)
        J2_tot = arakawa_jacobian(ψ2, s.q2_prime, dx, dy)
        
        diag.q1Jbar .+= vec(mean(s.q1_prime .* J1_tot, dims=1))
        diag.q2Jbar .+= vec(mean(s.q2_prime .* J2_tot, dims=1))
        
        diag.dy_v_qpsq1 .+= vec(d_dy(mean(v1 .* (s.q1_prime.^2), dims=1), dy))
        diag.dy_v_qpsq2 .+= vec(d_dy(mean(v2 .* (s.q2_prime.^2), dims=1), dy))
        
        diag.q1τ .+= vec(mean(v1 .* s.q1_prime, dims=1))
        diag.q2τ .+= vec(mean(v2 .* s.q2_prime, dims=1))
        
        diag.rq2ζ2 .+= vec(mean(r .* s.q2_prime .* o.L2D(s.ψ2_prime), dims=1))
        
        diag.counter += 1
    end
end

#########################################3
## Zonal-mean energy budget
mutable struct ZonalMeanEnergyDiag <: AbstractDiagnostic
    frequency::Int
    counter::Int
    times::Vector{Float64}
    
    # Data matrices: rows=y, cols=time
    CBC::Matrix{Float64}
    CBT::Matrix{Float64}
    therm_damping::Matrix{Float64}
    mech_damping::Matrix{Float64}
end

function ZonalMeanEnergyDiag(nt::Int, dt::Float64, frequency::Int, Ny::Int)
    n_diag = ceil(Int, nt / frequency)
    return ZonalMeanEnergyDiag(
        frequency, 1, zeros(n_diag),
        zeros(Ny, n_diag), zeros(Ny, n_diag), zeros(Ny, n_diag), zeros(Ny, n_diag)
    )
end

function compute!(diag::ZonalMeanEnergyDiag, model::QGModel, t::Float64, step::Int)
    if step % diag.frequency == 0
        idx = diag.counter
        s = model.state
        
        diag.times[idx] = t
        
        ψ1 = s.ψ1_bar' .+ s.ψ1_prime
        ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        
        u1_prime, v1_prime = u_from_psi(s.ψ1_prime)
        u2_prime, v2_prime = u_from_psi(s.ψ2_prime)
        
        u1t, v1t = u_from_psi(ψ1)
        u2t, v2t = u_from_psi(ψ2)
        
        u1_bar = mean(u1t, dims=1)
        u2_bar = mean(u2t, dims=1)
        
        # Populate matrices slice-by-slice
        diag.CBC[:, idx] .= vec((u1_bar .- u2_bar) .* mean(s.ψ1_prime .* v2_prime, dims=1) ./ 2)
        
        diag.CBT[:, idx] .= vec(u1_bar .* d_dy(mean(u1_prime .* v1_prime, dims=1), dy) .+ 
                                u2_bar .* d_dy(mean(u2_prime .* v2_prime, dims=1), dy))
        
        diag.therm_damping[:, idx] .= vec(-α * F1 .* ((s.ψ1_bar .- s.ψ2_bar) ./ 2))
        
        diag.mech_damping[:, idx] .= vec(-r .* u2_bar)
        
        diag.counter += 1
    end
end

#########################################3
## BC zone-averaged energy budget
mutable struct ScalarEnergyDiag <: AbstractDiagnostic
    frequency::Int
    counter::Int
    times::Vector{Float64}
    ind_start::Int
    ind_end::Int
    
    # Time series of domain-averaged scalars
    CBC::Vector{Float64}
    CBT::Vector{Float64}
    therm_damping::Vector{Float64}
    mech_damping::Vector{Float64}
end

function ScalarEnergyDiag(nt::Int, dt::Float64, frequency::Int, ind_start::Int, ind_end::Int)
    n_diag = ceil(Int, nt / frequency)
    return ScalarEnergyDiag(
        frequency, 1, zeros(n_diag), ind_start, ind_end,
        zeros(n_diag), zeros(n_diag), zeros(n_diag), zeros(n_diag)
    )
end

function compute!(diag::ScalarEnergyDiag, model::QGModel, t::Float64, step::Int)
    if step % diag.frequency == 0
        idx = diag.counter
        s = model.state
        
        diag.times[idx] = t
        
        ψ1 = s.ψ1_bar' .+ s.ψ1_prime
        ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        
        u1_prime, v1_prime = u_from_psi(s.ψ1_prime)
        u2_prime, v2_prime = u_from_psi(s.ψ2_prime)
        
        u1t, v1t = u_from_psi(ψ1)
        u2t, v2t = u_from_psi(ψ2)
        
        u1_bar = mean(u1t, dims=1)
        u2_bar = mean(u2t, dims=1)
        
        # Calculate intermediate 1D arrays
        temp_CBC = (u1_bar .- u2_bar) .* mean(s.ψ1_prime .* v2_prime, dims=1) ./ 2
        temp_CBT = u1_bar .* d_dy(mean(u1_prime .* v1_prime, dims=1), dy) .+ 
                   u2_bar .* d_dy(mean(u2_prime .* v2_prime, dims=1), dy)
        temp_therm = α * F1 .* mean(((s.ψ1_prime .- s.ψ2_prime).^2) ./ 2, dims=1)
        temp_mech = r .* mean(u2_prime.^2 .+ v2_prime.^2, dims=1)
        
        # Avert allocation by taking the mean over the specified index range
        idx_range = diag.ind_start:diag.ind_end
        
        diag.CBC[idx] = mean(temp_CBC[idx_range])
        diag.CBT[idx] = mean(temp_CBT[idx_range])
        
        # Original code subtracts these, so we store them as negative quantities to match
        diag.therm_damping[idx] = -mean(temp_therm[idx_range])
        diag.mech_damping[idx] = -mean(temp_mech[idx_range])
        
        diag.counter += 1
    end
end




################################################################################
## Zonal-Mean Flow Hovmoller Diagnostic
################################################################################
mutable struct HovmollerZonalFlowDiag <: AbstractDiagnostic
    frequency::Int
    counter::Int
    times::Vector{Float64}
    
    # 2D matrices: rows = y (space), columns = time
    U1::Matrix{Float64}
    U2::Matrix{Float64}
end

function HovmollerZonalFlowDiag(nt::Int, dt::Float64, frequency::Int, Ny::Int)
    n_diag = ceil(Int, nt / frequency)
    return HovmollerZonalFlowDiag(
        frequency, 
        1, 
        zeros(n_diag),
        zeros(Ny, n_diag), 
        zeros(Ny, n_diag)
    )
end

function compute!(diag::HovmollerZonalFlowDiag, model::QGModel, t::Float64, step::Int)
    if step % diag.frequency == 0
        idx = diag.counter
        s = model.state
        
        # Save current time
        diag.times[idx] = t
        
        # Calculate total streamfunction (background + perturbation)
        ψ1 = s.ψ1_bar' .+ s.ψ1_prime
        ψ2 = s.ψ2_bar' .+ s.ψ2_prime
        
        # Extract total zonal velocity (u)
        u1_total, _ = u_from_psi(ψ1)
        u2_total, _ = u_from_psi(ψ2)
        
        # Compute the zonal mean (averaging along x, keeping y)
        # vec() converts the 1 x Ny mean matrix into an Ny-element vector
        diag.U1[:, idx] .= vec(mean(u1_total, dims=1))
        diag.U2[:, idx] .= vec(mean(u2_total, dims=1))
        
        diag.counter += 1
    end
end


