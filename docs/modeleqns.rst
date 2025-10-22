=========================================
Model equations
=========================================
Here we provide that governing equations of the two-layer QG channel model.

Governing equations
~~~~~~~~~~~~~~~~~~~~~~~~~
The governing equations for the present two-layer QG model are

.. math::

   \begin{equation}
   \partial_{t} q_{1}
   + \mathrm{J} (\psi_{1}, \, q_{1} )
   = \kappa_\mathrm{T} \left (
   \frac{\psi_{1} - \psi_{2}}{2} - \tau_\mathrm{eq.}
   \right )
   - \nu \nabla^4 q_{1}
   \end{equation}
.. math::
   \begin{equation}
   \partial_{t} q_{2}
   + \mathrm{J} (\psi_{2}, \, q_{2} )
   = - \kappa_\mathrm{T} \left (
   \frac{\psi_{1} - \psi_{2}}{2} - \tau_\mathrm{eq.}
   \right )
   - \mu \nabla^2 \psi_{2}
   - \nu \nabla^4 q_{2}
   \end{equation}

Linear friction in the lower layer has the coefficient
:math:`\mu`, with dimensions of inverse time.
Hyperviscosity is applied to the streamfunction in both layers
and has the coefficient :math:`\nu` with dimensions of :math:`\mathrm{L}^{4} \mathrm{T}`.
:math:`\kappa_\mathrm{T}` is an inverse time scale for damping the baroclinic
flow to the background streamfunction, :math:`\tau_\mathrm{eq.}`.


We now describe a specific example.
