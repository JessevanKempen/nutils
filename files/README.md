Graduation project: Quantifying uncertainty in the subsurface of geothermal systems  {#mainpage}
============================

This repository contains a geothermal doublet model to be used for uncertainty quantification based on real-world data

### Main program
The main program can be found in the **myMain.py** file and can be executed using Python 3.5. The environment required
is found in **environment.yml**.


### Modules
The following modules are used for this finite element program:
- **myFEM**	        This module contains the analytical model and the finite element thermo-hydraulic model
- **myIOlib**	    This module contains a json parameter creator and basic plotting function
- **myUQ**	        This module contains the forward uncertainty quantification

### Supplementary data
- **output** The default (empty) output directory

### Documentation
Documentation is available in the **doc** directory:
- **html/index.html** can be opened to browse the documentation
- **latex/refman.pdf** contains the LaTeX generated reference manual