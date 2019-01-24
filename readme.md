# Densitymap

Generate 6D particle distributions from image files for use in Astra particle tracking code

## Getting started

Clone and run densitymap.py

### Prerequisites

Numpy

pyDOE						: Latin hypercube 6D random number generation

ghalton						: Pseudo random number generation in the Halton sequence

scipy.interpolate			: Interpolating when sampling CDF

matplotlib.pyplot			: Image reading

## Example

Generate 100000 particles from 2d transverse distrbution in variable pic, with 
pixel resolution 20 um, 200 pC total charge, and Fermi Dirac momentum distrbution.

Save to file "image_dist_100k.ini"

```
dm = DensityMap()
dm.set_transverse_image(pic, 20e6)
dm.set_charge(200e-12)
dm.set_momentum_fermi_dirac(4.71, 4.46, 0)
dm.save_astra_distrbution("image_dist_100k.ini", 100e3)
```