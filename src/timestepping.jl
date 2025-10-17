
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
