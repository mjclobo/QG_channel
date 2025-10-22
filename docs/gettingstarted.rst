=========================================
Getting started
=========================================

Installing Julia
~~~~~~~~~~~~~~~~~~~~~~~~~
For official instructions on instaling Julia,
please refer to `this website<https://julialang.org/install/>`_.


Work environment
~~~~~~~~~~~~~~~~~~~~~~~~~~
I use the `VS Code<https://code.visualstudio.com/download>`_ code
editor/graphical user interface.

The Jupyter interactive environment is another great option!

Required packages
~~~~~~~~~~~~~~~~~~~~
You can download the required packages with the command


.. code-block:: julia

   using Pkg
   Pkg.add(["FFTW", "LinearAlgebra", "Statistics", "Dates", "JLD2", "PyPlot", "Printf", "Random", "SparseArrays", "LoopVectorization"])


If you plan on using Julia more (or already do), I recommend creating a project environment.
I will add details on how to do this in the future.


