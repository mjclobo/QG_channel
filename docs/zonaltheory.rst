=========================================
Background math
=========================================

On this page we provide the math background required
to understand how CWT_Multi, and wavelet transforms more generally,
work.
We present ideas and illustrations, rather than comprehensive maths.
Future pages in this documentation will cover the mathematical details.
For now, we refer the reader to Lobo et al. (2024), and include
additional references at the bottom of this page.


The following sections aim to be self-contained and therefore contain
information of increasing level of complexity.
The reader is encouraged to scan the material until they
come to a section that contains information unfamiliar
to them.
If the reader is familiar with the basics of signal processing, then
they may wish to proceed directly to the *Basic CWT_Multi theory* page.

Sinusoids and complex exponentials
~~~~~~~~~~~~~~~~~~~~~~~~~
A *sine wave* is a signal that takes the form

   .. math::
    s(t) = A \mathrm{sin} ( \omega t + \phi ) \, ,

where

- :math:`A` is the *amplitude* of the wave with the units of your signal (e.g., meters for water level data)
- :math:`\omega = 2 \pi f` is the *angular frequency* with units rad/s
- :math:`f` is the *frequency* with units of cycles/s, i.e., Hz
- :math:`t` is time with units of seconds
- :math:`\phi` is a phase offset

The total argument to the sine function is known as the *phase*,
a unitless quantity that determines the output from the sine function
(a value in the range :math:`-1` to :math:`1`).

The Fourier transform, in pictures
~~~~~~~~~~~~~~~~~~~~~~~~~
The *Fourier transform* provides the means to decompose a
signal into a linear superposition (i.e., a sum) of sine
waves.
In particular, a Fourier transform routine will provide the user
with the amplitude, frequency and phase offset for each sine wave
that contributes to the original signal.
Though there are many technical aspects to both the mathematics
and the practical application of Fourier transforms, the information
presented so far is sufficient for our purposes.


.. image:: /images/FT_drawing.png
   :width: 300pt




Additional reading
~~~~~~~~~~~~~~~~~~~~~~~~~
- We recommend `this 3Blue1Brown video <https://www.youtube.com/watch?v=spUNpyF58BY>`_
  for an intuitive introduction to the Fourier Transform.
- Jonathan Lilly has great `course material <http://jmlilly.net/course/index.html>`_
  for more details on wavelet analysis and
  signal processing, more generally.


