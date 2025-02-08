# The replications algorithm for Discrete-Event Simulation

The materials in this repo provide an implementation of the Replications Algorithm to automatically select the no. of replications for a Discrete-Event Simulation.

The main tutorial can be found in the [automated_reps.ipynb](./automated_reps.ipynb) notebook.

## License

The materials have been made available under an [MIT license](LICENCE).  The materials are as-is with no liability for the author. Please provide credit if you reuse the code in your own work.

## Citation

Please feel free to use or adapt the code for your own work. But if so then a citation would be very much appreciated! 

```bibtex
@software{The_replications_algorithm,
author = {Monks, Thomas },
license = {MIT},
title = {{The replications algorithm for DES in Python}},
url = {https://github.com/TheOpenScienceNerd/replications-algorithm}
}
```

## Installation instructions

### Installing dependencies

All dependencies can be found in [`binder/environment.yml`]() and are pulled from conda-forge.  To run the code locally, we recommend installing [miniforge](https://github.com/conda-forge/miniforge);

> miniforge is Free and Open Source Software (FOSS) alternative to Anaconda and miniconda that uses conda-forge as the default channel for packages. It installs both conda and mamba (a drop in replacement for conda) package managers.  We recommend mamba for faster resolving of dependencies and installation of packages. 

navigating your terminal (or cmd prompt) to the directory containing the repo and issuing the following command:

```bash
mamba env create -f binder/environment.yml
```

Activate the mamba environment using the following command:

```bash
mamba activate rep_alg
```

Run Jupyter-lab

```bash
jupyter-lab
```

## Repo overview

```
.
├── binder
│   └── environment.yml
├── callcentresim
│   ├── __init__.py
│   ├── model.py
│   └── output_analysis.py
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── images
│   └── ...
├── automated_reps.ipynb
├── README.md
└── rep_utility.py
```

* `environment.yml` - contains the conda environment if you wish to work locally algorithm
* `automated_reps` - main notebook file containing the tutorial code for the replications algorithm
* `rep_utility` - local python module with supporting code for the algorithm
* `callcentresim` - local python package containing the urgent care call centre SimPy model.
* `images` -images for the notebook
* `CHANGES.md` - changelog with record of notable changes to project between versions.
* `CITATION.cff` - citation information for the package.
* `LICENSE` - details of the MIT permissive license of this work.



