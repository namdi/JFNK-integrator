# Jacobian-Freen Newton Krylov Method
This is a git repository for the code for a group of numerical integrators for time-dependent differential equations. They include the backward (implicit) Euler method, Spectral Deferred Corrections (SDC) method*, and a *Jacobian-Free Newton Krylov (JFNK) method** designed to acclearate the convergence of SDC for **stiff** systems.

 
A mathematical description/ theory behind the numerical integrators are in a .pdf file called math_nodes.pdf. I suggest reading that if you are interested in understanding the mechanics behind the algorithms.

Code documentation and the users guide for the JFNK may be found in the following 
formats and locations:
 
    1. .pdf format: docs\_build\latex\JFNK.tex
    2. .html format: docs\_build\html\index.html