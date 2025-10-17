################################################################################
# define constants
################################################################################



################################################################################
# define grid etc.
################################################################################

# Get spectral multipliers for rfft in x
function get_kx_rfft(Nx, Lx)
    return 2π * collect(0:(Nx÷2)) / Lx
end

# Wavenumbers for full FFT in x (length Nx)
# kx = 2π * vcat(0:Nx÷2-1, -Nx÷2:-1) / Lx

# Wavenumbers for rfft in x (length Nx/2+1)
kx_spec = get_kx_rfft(Nx, Lx)  # (Nx/2+1)

# Wavenumbers in y for FD -- y has Dirichlet BC, so modes are sines:
# For spectral operators in y (if needed), ky = π * (1:Ny) / Ly
# but here finite differences in y, so no spectral ky needed.

# Correct kx definition for full FFT (length Nx)
kx = 2π * vcat(0:Nx÷2, -Nx÷2+1:-1) / Lx  # length Nx

# KX shape must be (Nx, Ny) to match data shape (Nx, Ny)
KX = repeat(kx, 1, Ny)  # size (Nx, Ny)
KXr = get_kx_rfft(Nx, Lx)

# Fourier wavenumbers for rfft
nkx = Nx ÷ 2 + 1
kx = 2π * vcat(0:Nx÷2) / (dx * Nx)
k2 = reshape(kx.^2, nkx, 1)[:]

