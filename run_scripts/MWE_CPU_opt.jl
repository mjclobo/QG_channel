Nx = 32
Ny = 64
dt = 0.01
nt = 5

rhs_ws = RHSWorkspace(Nx, Ny)
rk4_ws = RK4Workspace(Nx, Ny, rhs_ws)

q1_bar = zeros(Ny)
q2_bar = zeros(Ny)
q1_prime = 1e-2 * randn(Nx, Ny)
q2_prime = 1e-2 * randn(Nx, Ny)
t0 = 0.0
params = ModelParams(Nx, Ny, nt, Lx, Ly, dt, beta, f0, g, [H1,H2], ρ0, Δρ, ν, r, α, U0, WC)

run_model_decomp_alloc!(q1_bar, q2_bar, q1_prime, q2_prime, t0, params;
                        nt=nt, dt=dt, rk4_ws=rk4_ws, output_every=1)