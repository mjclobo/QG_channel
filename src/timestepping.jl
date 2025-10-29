
################################################################################
# RK4 time stepper
################################################################################

function rk4(q1, q2, dt)
    k1q1, k1q2 = rhs(q1, q2)
    k2q1, k2q2 = rhs(q1 .+ 0.5dt .* k1q1, q2 .+ 0.5dt .* k1q2)
    k3q1, k3q2 = rhs(q1 .+ 0.5dt .* k2q1, q2 .+ 0.5dt .* k2q2)
    k4q1, k4q2 = rhs(q1 .+ dt .* k3q1, q2 .+ dt .* k3q2)

    q1_new = q1 .+ dt/6 .* (k1q1 .+ 2k2q1 .+ 2k3q1 .+ k4q1)
    q2_new = q2 .+ dt/6 .* (k1q2 .+ 2k2q2 .+ 2k3q2 .+ k4q2)
    return q1_new, q2_new
end

function rk4_prime(q1_prime, q2_prime, q1_bar, q2_bar, dt)
    k1q1, k1q2 = rhs_prime(q1_prime, q2_prime, q1_bar, q2_bar)
    k2q1, k2q2 = rhs_prime(q1_prime .+ 0.5dt .* k1q1, q2_prime .+ 0.5dt .* k1q2, q1_bar, q2_bar)
    k3q1, k3q2 = rhs_prime(q1_prime .+ 0.5dt .* k2q1, q2_prime .+ 0.5dt .* k2q2, q1_bar, q2_bar)
    k4q1, k4q2 = rhs_prime(q1_prime .+ dt .* k3q1, q2_prime .+ dt .* k3q2, q1_bar, q2_bar)

    q1_new = q1_prime .+ dt/6 .* (k1q1 .+ 2k2q1 .+ 2k3q1 .+ k4q1)
    q2_new = q2_prime .+ dt/6 .* (k1q2 .+ 2k2q2 .+ 2k3q2 .+ k4q2)
    return q1_new, q2_new
end

function rk4_bar(q1_bar, q2_bar, q1_prime, q2_prime, dt)
    k1q1, k1q2 = rhs_bar(q1_bar, q2_bar, q1_prime, q2_prime)
    k2q1, k2q2 = rhs_bar(q1_bar .+ 0.5dt .* k1q1, q2_bar .+ 0.5dt .* k1q2, q1_prime, q2_prime)
    k3q1, k3q2 = rhs_bar(q1_bar .+ 0.5dt .* k2q1, q2_bar .+ 0.5dt .* k2q2, q1_prime, q2_prime)
    k4q1, k4q2 = rhs_bar(q1_bar .+ dt .* k3q1, q2_bar .+ dt .* k3q2, q1_prime, q2_prime)

    q1_new = q1_bar .+ dt/6 .* (k1q1 .+ 2k2q1 .+ 2k3q1 .+ k4q1)
    q2_new = q2_bar .+ dt/6 .* (k1q2 .+ 2k2q2 .+ 2k3q2 .+ k4q2)
    return q1_new, q2_new
end

function rk4_coupled(q1_prime, q2_prime, q1_bar, q2_bar, dt)
    """
    Fully coupled RK4 step for (q_prime, q_bar) system.
    
    Parameters
    ----------
    q1_prime, q2_prime : arrays
        Perturbation PV at current time.
    q1_bar, q2_bar : arrays
        Zonal-mean PV at current time.
    dt : float
        Timestep.
    
    Returns
    -------
    Updated (q1_prime, q2_prime, q1_bar, q2_bar) after one dt
    """
    
    # Stage 1
    k1p1, k1p2 = rhs_prime(q1_prime, q2_prime, q1_bar, q2_bar)
    k1b1, k1b2 = rhs_bar(q1_bar, q2_bar, q1_prime, q2_prime)
    
    # Stage 2
    q1p_temp = q1_prime + 0.5*dt*k1p1
    q2p_temp = q2_prime + 0.5*dt*k1p2
    q1b_temp = q1_bar   + 0.5*dt*k1b1
    q2b_temp = q2_bar   + 0.5*dt*k1b2
    k2p1, k2p2 = rhs_prime(q1p_temp, q2p_temp, q1b_temp, q2b_temp)
    k2b1, k2b2 = rhs_bar(q1b_temp, q2b_temp, q1p_temp, q2p_temp)
    
    # Stage 3
    q1p_temp = q1_prime + 0.5*dt*k2p1
    q2p_temp = q2_prime + 0.5*dt*k2p2
    q1b_temp = q1_bar   + 0.5*dt*k2b1
    q2b_temp = q2_bar   + 0.5*dt*k2b2
    k3p1, k3p2 = rhs_prime(q1p_temp, q2p_temp, q1b_temp, q2b_temp)
    k3b1, k3b2 = rhs_bar(q1b_temp, q2b_temp, q1p_temp, q2p_temp)
    
    # Stage 4
    q1p_temp = q1_prime + dt*k3p1
    q2p_temp = q2_prime + dt*k3p2
    q1b_temp = q1_bar   + dt*k3b1
    q2b_temp = q2_bar   + dt*k3b2
    k4p1, k4p2 = rhs_prime(q1p_temp, q2p_temp, q1b_temp, q2b_temp)
    k4b1, k4b2 = rhs_bar(q1b_temp, q2b_temp, q1p_temp, q2p_temp)
    
    # Combine stages
    q1_prime_new = q1_prime + dt/6 .* (k1p1 + 2*k2p1 + 2*k3p1 + k4p1)
    q2_prime_new = q2_prime + dt/6 .* (k1p2 + 2*k2p2 + 2*k3p2 + k4p2)
    q1_bar_new   = q1_bar   + dt/6 .* (k1b1 + 2*k2b1 + 2*k3b1 + k4b1)
    q2_bar_new   = q2_bar   + dt/6 .* (k1b2 + 2*k2b2 + 2*k3b2 + k4b2)
    
    return q1_prime_new, q2_prime_new, q1_bar_new, q2_bar_new
end

################################################################################
# RK4 time stepper w/ integrating factor
################################################################################

function nonlinear_rhs(q1, q2)
    ψ1, ψ2 = invert_qg_pv(q1, q2)

    J1 = arakawa_jacobian(ψ1, q1)
    J2 = arakawa_jacobian(ψ2, q2)

    dq1dt = -J1 .- beta .* u_from_psi(ψ1)[2]
    dq2dt = -J2 .- beta .* u_from_psi(ψ2)[2]

    return dq1dt, dq2dt
end

"""
    rk4_integrating_factor(q1, q2, dt)

Time steps the QG PV fields using RK4 for nonlinear terms and exact integration for linear terms.
"""
function rk4_integrating_factor(q1, q2, dt)

    Nx, Ny = size(q1)
    Nkx = div(Nx, 2) + 1  # rfft size

    # FFT of input
    q1_hat = rfft(q1, 1)
    q2_hat = rfft(q2, 1)

    # Allocate working arrays
    N1_0 = similar(q1)
    N2_0 = similar(q2)
    N1_1 = similar(q1)
    N2_1 = similar(q2)
    N1_2 = similar(q1)
    N2_2 = similar(q2)
    N1_3 = similar(q1)
    N2_3 = similar(q2)

    # Stage 1
    N1_0, N2_0 = nonlinear_rhs(q1, q2)

    # Stage 2
    @turbo for j in 1:Ny, i in 1:Nx
        N1_1[i,j] = q1[i,j] + 0.5 * dt * N1_0[i,j]
        N2_1[i,j] = q2[i,j] + 0.5 * dt * N2_0[i,j]
    end
    N1_1, N2_1 = nonlinear_rhs(N1_1, N2_1)

    # Stage 3
    @turbo for j in 1:Ny, i in 1:Nx
        N1_2[i,j] = q1[i,j] + 0.5 * dt * N1_1[i,j]
        N2_2[i,j] = q2[i,j] + 0.5 * dt * N2_1[i,j]
    end
    N1_2, N2_2 = nonlinear_rhs(N1_2, N2_2)

    # Stage 4
    @turbo for j in 1:Ny, i in 1:Nx
        N1_3[i,j] = q1[i,j] + dt * N1_2[i,j]
        N2_3[i,j] = q2[i,j] + dt * N2_2[i,j]
    end
    N1_3, N2_3 = nonlinear_rhs(N1_3, N2_3)

    # Combine RK4 increments
    @turbo for j in 1:Ny, i in 1:Nx
        N1_0[i,j] = (N1_0[i,j] + 2*N1_1[i,j] + 2*N1_2[i,j] + N1_3[i,j]) / 6
        N2_0[i,j] = (N2_0[i,j] + 2*N2_1[i,j] + 2*N2_2[i,j] + N2_3[i,j]) / 6
    end

    # FFT of nonlinear terms
    N1_hat = rfft(N1_0, 1)
    N2_hat = rfft(N2_0, 1)

    # Prepare output arrays (overwrite q1_hat/q2_hat)
    Threads.@threads for i in 1:Nkx
        L_kx = L_ops[i]  # precomputed exp(dt*A_kx), size 2Ny x 2Ny

        qvec = @view q1_hat[i, :]
        qvec2 = @view q2_hat[i, :]
        Nvec = @view N1_hat[i, :]
        Nvec2 = @view N2_hat[i, :]

        # Combine into single vectors
        qin = Vector{ComplexF64}(undef, 2Ny)
        Nin = Vector{ComplexF64}(undef, 2Ny)

        @inbounds @simd for j in 1:Ny
            qin[j] = qvec[j]
            qin[Ny + j] = qvec2[j]
            Nin[j] = Nvec[j]
            Nin[Ny + j] = Nvec2[j]
        end

        # Apply integrating factor
        qout = L_kx * qin .+ dt .* Nin

        # Split back
        @inbounds @simd for j in 1:Ny
            qvec[j] = qout[j]
            qvec2[j] = qout[Ny + j]
        end
    end

    # Inverse FFT to get updated q1, q2
    q1_new = irfft(q1_hat, Nx, 1)
    q2_new = irfft(q2_hat, Nx, 1)

    return q1_new, q2_new
end
