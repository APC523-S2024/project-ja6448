# Final Project - Numerically solving the Allen-Cahn equation
Final Project - Josh Arrington, Spring 2024
In this project, I numerically solved the Allen-Cahn equation in 1D and 2D using techniques we discussed in lecture.
The Allen-Cahn equation is a nonlinear PDE of the form $$ \tau \frac{\partial \phi}{\partial t} = -\frac{\delta F[\phi]}{\delta \phi} $$

# Generating the data used in the report
This project was written in Python 3.12.2 with the popular numerical libraries Numpy (1.26.4), Scipy (1.13.0), and Matplotlib (3.8.4). The necessary simulation files are in the "src" directory.
The built-in Python modules available on Adroit do not have the correct versioning for this, and may throw an error for the 2D simulations (namely, in how I calculate the contour levels). 
I made the decision to use the updated version because Matplotlib's contour `collections` attribute is deprecated and will be removed soon (but the `get_paths()` method is not available in the older versions of Matplotlib like what are available in Adroit).
The simulations run fine on Adroit, but the error analysis in 2D is what necessitates updating Matplotlib in a custom `conda` environment.

# Example Results
Several videos are available in the "slides" directory, and figures are available in the "figures" directory.
