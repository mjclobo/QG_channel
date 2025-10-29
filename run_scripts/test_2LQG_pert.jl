using SparseArrays, LinearAlgebra
using Base.Threads
using LoopVectorization

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
Nx = 128
Ny = 128
Lx = 1.0
Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny

F1 = 1.0
F2 = 1.0
dt = 0.01
τ = 1.0   # relaxation timescale for damping to background
Nt = 100  # number of time steps

# ------------------------------------------------------------
# Indexing
# ------------------------------------------------------------
idx(i, j, Nx) = (j - 1) * Nx + i

# ------------------------------------------------------------
# Laplacian with Dirichlet in y, periodic in x
# ------------------------------------------------------------
function build_sparse_laplacian_dirichlet_y(Nx, Ny, dx, dy)
    N = Nx * Ny
    D = spzeros(N, N)
    for j in 1:Ny
        for i in 1:Nx
            n = idx(i,j,Nx)
            if j==1 || j==Ny
                D[n,n] = 1.0
                continue
            end
            D[n,n] = -2/dx^2 - 2/dy^2
            D[n, idx(mod1(i-1,Nx), j,Nx)] += 1/dx^2
            D[n, idx(mod1(i+1,Nx), j,Nx)] += 1/dx^2
            D[n, idx(i,j-1,Nx)] += 1/dy^2
            D[n, idx(i,j+1,Nx)] += 1/dy^2
        end
    end
    return D
end

# ------------------------------------------------------------
# Precompute inversion operator for δψ
# ------------------------------------------------------------
function precompute_operators_delta(Nx, Ny, F1, F2, dx, dy)
    N = Nx*Ny
    L = build_sparse_laplacian_dirichlet_y(Nx, Ny, dx, dy)
    I_N = spdiagm(0=>ones(N))
    A11 = L - F1*I_N
    A12 = F1*I_N
    A21 = F2*I_N
    A22 = L - F2*I_N
    for j in (1,Ny)
        for i in 1:Nx
            n = idx(i,j,Nx)
            A11[n,:] .= 0.0; A11[n,n] = 1.0; A12[n,:] .= 0.0
            A22[n,:] .= 0.0; A22[n,n] = 1.0; A21[n,:] .= 0.0
        end
    end
    A = [A11 A12; A21 A22]
    return lu(A)
end

# ------------------------------------------------------------
# Laplacian using LoopVectorization
# ------------------------------------------------------------
function laplacian!(Δψ, ψ, Nx, Ny, dx, dy)
    @tturbo for j in 2:Ny-1
        for i in 1:Nx
            ip = mod1(i+1, Nx)
            im = mod1(i-1, Nx)
            Δψ[i,j] = (ψ[ip,j] - 2ψ[i,j] + ψ[im,j])/dx^2 +
                      (ψ[i,j+1] - 2ψ[i,j] + ψ[i,j-1])/dy^2
        end
    end
end

# ------------------------------------------------------------
# Invert δq to δψ
# ------------------------------------------------------------
function invert_qg_pv_delta(δq1, δq2, Nx, Ny, inversion_ops)
    N = Nx*Ny
    rhs = vcat(vec(δq1), vec(δq2))
    δψ_vec = inversion_ops \ rhs
    δψ1 = reshape(δψ_vec[1:N], Nx, Ny)
    δψ2 = reshape(δψ_vec[N+1:end], Nx, Ny)
    δψ1[:,1] .= 0.0; δψ1[:,end] .= 0.0
    δψ2[:,1] .= 0.0; δψ2[:,end] .= 0.0
    return δψ1, δψ2
end

# ------------------------------------------------------------
# Invert PV to ψ with evolving background
# ------------------------------------------------------------
function invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, Nx, Ny, F1, F2, inversion_ops)
    Δψ1_bg = zeros(Nx, Ny)
    Δψ2_bg = zeros(Nx, Ny)
    laplacian!(Δψ1_bg, ψ1_bg, Nx, Ny, dx, dy)
    laplacian!(Δψ2_bg, ψ2_bg, Nx, Ny, dx, dy)

    δq1 = q1 - (Δψ1_bg - F1*(ψ1_bg - ψ2_bg))
    δq2 = q2 - (Δψ2_bg - F2*(ψ2_bg - ψ1_bg))

    δψ1, δψ2 = invert_qg_pv_delta(δq1, δq2, Nx, Ny, inversion_ops)

    ψ1 = ψ1_bg + δψ1
    ψ2 = ψ2_bg + δψ2

    return ψ1, ψ2
end

# ------------------------------------------------------------
# Update background streamfunction
# ------------------------------------------------------------
function update_background!(q1_bg, q2_bg, ψ1_bg, ψ2_bg, Nx, Ny, F1, F2, dt, τ)
    # Relaxation (optional forcing)
    @threads for j in 1:Ny
        for i in 1:Nx
            ψ1_bg[i,j] += dt/τ * (ψ1_bg[i,j] - ψ1_bg[i,j])
            ψ2_bg[i,j] += dt/τ * (ψ2_bg[i,j] - ψ2_bg[i,j])
        end
    end

    # Compute updated ψ_bg from q_bg (Poisson solve approximation)
    Δψ1_bg = zeros(Nx, Ny)
    Δψ2_bg = zeros(Nx, Ny)
    laplacian!(Δψ1_bg, ψ1_bg, Nx, Ny, dx, dy)
    laplacian!(Δψ2_bg, ψ2_bg, Nx, Ny, dx, dy)

    ψ1_bg .= (q1_bg + F1*ψ2_bg + Δψ1_bg) ./ (1 + F1)
    ψ2_bg .= (q2_bg + F2*ψ1_bg + Δψ2_bg) ./ (1 + F2)
end

# ------------------------------------------------------------
# Initialize fields
# ------------------------------------------------------------
ψ1_bg = zeros(Nx, Ny)
ψ2_bg = zeros(Nx, Ny)
q1_bg = zeros(Nx, Ny)
q2_bg = zeros(Nx, Ny)

ψ1 = copy(ψ1_bg)
ψ2 = copy(ψ2_bg)
q1 = copy(q1_bg)
q2 = copy(q2_bg)

# Inversion operator
inversion_ops = precompute_operators_delta(Nx, Ny, F1, F2, dx, dy)

# ------------------------------------------------------------
# Time-stepping loop
# ------------------------------------------------------------
for n in 1:Nt
    # 1. Update ψ_bg from q_bg
    update_background!(q1_bg, q2_bg, ψ1_bg, ψ2_bg, Nx, Ny, F1, F2, dt, τ)

    # 2. Compute new ψ from current q and evolving background
    ψ1, ψ2 = invert_qg_pv(q1, q2, ψ1_bg, ψ2_bg, Nx, Ny, F1, F2, inversion_ops)

    # 3. Update PV q1, q2 here according to your dynamics
    # For example, simple damping to background:
    @threads for j in 1:Ny
        for i in 1:Nx
            q1[i,j] += dt/τ*(q1_bg[i,j] - q1[i,j])
            q2[i,j] += dt/τ*(q2_bg[i,j] - q2[i,j])
        end
    end

    # Optional: output, diagnostics, etc.
end
