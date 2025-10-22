=========================================
Code and numerics
=========================================
The model is gridded in both $x$ and $y$, though spectral methods
are often used for calculations in the $x$ direction.
In the near future I plan to provide the option of making the model
completely spectral in the $x$ direction, so that, for example,
one could isolate differents scales of zonal wave-mean flow interaction.


Timestepper
~~~~~~~~~~~~~~~~~~~~~~~~~
RK4 (switching to integrating factor for linear terms is on the short list)


Boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~
We apply no normal flow boundary conditions
on the meridional walls.
In addition, we apply the no stress condition ($\nabla^2 \psi = 0$)
in order to satisfy the boundary condition for the biharmonic viscosity.


General flow of code
~~~~~~~~~~~~~~~~~~~~~~~~~
The code follows a standard workflow for a QG model.
We time step the QG PV where the right hand side of the time
tendency equation is calculated in rhs() function in teh 2LQG_main.jl file.
This righthand side includes (in order)

- Invert QG PV to get streamfunction
- Calculate Jacobian term
- Add $-\beta v_{k}$ to the time tendency of QG PV
- Add hyperviscosity
- Add bottom friction
- Apply thermal damping

See the rhs() function for more details.



