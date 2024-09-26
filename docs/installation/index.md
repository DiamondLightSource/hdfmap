# Installation

HdfMap can be installed as a module in any python environment with python version > 3.10. 
The module is designed to be used either in an interactive environment or in python scripts.

*Requires:* Python >=3.10, Numpy, h5py


=== "PyPi"

    ### pip installation from stable version
    ```bash
    python -m pip install --upgrade hdfmap
    ```

=== "GitHub"

    ### pip installation from github channel
    ```bash
    python -m pip install --upgrade git+https://github.com/DiamondLightSource/hdfmap.git
    ```

=== "Conda"

    ### Full installation of Python environment using conda miniforge
    See [conda-forge](https://github.com/conda-forge/miniforge)
    #### Install miniforge (any conda env will do)
    ```bash
    cd location/of/miniforge
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-Linux-x86_64.sh
    ```
    You will be asked to enter the location to install conda and whether to change terminal commands [yes]. 
    
    #### Install HdfMap
    Then, in a new terminal:
    ```bash
    conda create -n hdfmapenv python
    conda activate hdfmapenv
    (hdfmapenv)$ python -m pip install hdfmap
    ```
