# TT-IGA

This repository contains the code for the paper ["Tensor train based isogeometric analysis for PDE approximation on parameter dependent geometries"](https://arxiv.org/pdf/2204.02843.pdf).

## Installation

### Requirements

For using just the TT-IGA package:

 * `pytorch>=1.7`
 * `numpy>=1.18`
 * `scipy`
 * [`torchtt`](https://github.com/ion-g-ion/torchtt)
 * `opt_einsum`
 * `matplotlib`
 * `pandas`

Optional dependencies to run the reference FEM solver are:
 * `fenics`
 
### Using pip

You can install the package using the following `pip` command after cloning the repository:

```
pip install .
```

## Packages

## Scripts and examples:


* [convergence_one_param.py](./examples/convergence_one_param.py): Convergence study  (**Test case 1**).
* [Cshape_study.py](./examples/Cshape_sudy.py): Increasing the number of parameters  (**Test case 2**).
* [material_jump.py](./examples/material_jump.py) / [material_jump.ipynb](./examples/material_jump.ipynb): Laplace equation in a cylinder with material filling  -> [Open using Google Colab](https://colab.research.google.com/github/ion-g-ion/coda-paper-tt-iga/blob/main/examples/material_jump.ipynb).
* [material_jump_study.py](./examples/material_jump_study.py) / [material_jump_study.ipynb](./examples/material_jump_study.ipynb): Laplace equation in a cylinder with material filling (**Test case 3**: tables).
* [waveguide.py](./examples/waveguide.py) / [waveguide.ipynb](./examples/waveguide.ipynb): Helmholtz equation inside a waveguide (**Test case 4**) -> [Open using Google Colab](https://colab.research.google.com/github/ion-g-ion/coda-paper-tt-iga/blob/main/examples/waveguide.ipynb).
* [twisted.py](./examples/twisted.py) / [twisted.ipynb](./examples/twisted.ipynb):


 The documentation is generated using `pdoc3` with:

 ```
 pdoc3 --html tt_iga -o docs/ --config latex_math=True --force
 ```

## Author

Ion Gabriel Ion, ion.ion.gabriel@gmail.com