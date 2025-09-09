# Emergent Periodicity in Fireflies

This repository provides supporting data and code for the eLife article:

Sarfati R, Joshi K, Martin O, Hayes JC, Iyer-Biswas S, Peleg O (2023)
Emergent periodicity in the collective synchronous flashing of fireflies
eLife 12:e78908. https://doi.org/10.7554/eLife.78908

----------------------------------------------------------------
### Folder structure

- EmergentPeriodicity/
  - Experimental_Results/
    - Original/
    - Corrected/
  - Simulation_Results/
    - Original/
    - Corrected/
  - Simulation_Code/
  - Theory_Results/
    - Original/
    - Corrected/

----------------------------------------------------------------
### Experimental_Results 
Relationship to the correction document tables: 
Table S1 (corrected dataset) lists every trial file present in Experimental_Results/Corrected.
Table S2 (original dataset) lists every trial file present in Experimental_Results/Original.

#### Folder Structure and File Description:
1. Folder "Xff":
Definition: "X" represents the number of fireflies involved in each experiment. This can be 1, 5, 10, 15, or 20. 
Contents: This folder contains two types of files:
1a. Interburst Interval Files (ibs): These files are prefixed with “ibs” and contain data on the interburst intervals, reported in seconds.
1b. Number of Flashes Time Series Files: Prefixed with “ns”, these files present a time series analysis. The first column indicates the time in seconds, and the second column enumerates the number of flashes. These are curated time series, corresponding to specific trials.

2. Folder "tsfigs":
Purpose: This folder is included for transparency purposes.
Contents: It contains raw time-series figures from the experiments, prior to any data trimming.
Data Representation: The figures depict time on the x-axis (in seconds) and the number of flashes on the y-axis.
Additional Note: Be aware that these raw data sets of the entire experiment recordings, may include anomalies such as significant light pollution, particularly when the tent was opened at the start and end of each experiment.

3. File starting with “envelope_01ff".
Definition: Contains data points for the envelope of the interburst interval (Tb) probability distribution function of a single firefly, for both the original (envelope_01ff_original.csv) and corrected (envelope_01ff_corrected.csv) datasets.Contents:  Each CSV file has two columns: the first lists Tb values in seconds, and the second lists the corresponding probability values.

4. File Naming Convention:
Structure = <MMDDYYYY><Camera><optional _n>
Date (MMDDYYYY) – month, day, and four-digit year of the trial.
Camera – single lowercase letter for the camera set (a, u, or c).
_n – underscore plus an integer (_1, _2, …) if the same camera set was restarted later that night.

Examples:
"06032020a_1" denotes data from June 3, 2020, captured by camera set "a", for the first time that night. 
"06032020a_2" denotes data from June 3, 2020, captured by camera set "a", for the second time that night. 
"06072020u" represents data from June 7, 2020, and was captured by camera set “u”.


----------------------------------------------------------------
###  Theory_Results
The Theory_Results folder contains the analytical model outputs generated from the single-firefly envelope. Each top-level folder (Original and Corrected) has two kinds of CSV files:

1. Interburst interval distributions 

- Filename format: (ibsXffTheory.csv)

- Definition: "X" represents the number of fireflies involved in each experiment. This can be 1, 5, 10, 15, or 20. 

- Columns:
    - Column 1 Interburst interval Tb [s]
    - Column 2 Probability density p(Tb)

2. Group-size standard deviations (sdTheory.csv)

- Filename: sdTheory.csv (one per folder).

- Columns:
    - Column 1 Number of fireflies (5, 10, 15, 20)

    - Column 2 Analytical standard deviation σ(Tb) [s]

#### Notes

- “Original” uses the envelope derived from the unedited single-firefly data.
- “Corrected” uses the envelope after trimming and removal.
- These theory tables were used to update Figures 3, 5, 7 in the main text and the reproduced Figures S3–S5 in the Supplementary Information.

----------------------------------------------------------------
###  Simulation_Results
This folder contains the results of the computational model. 
- Filename format: Interburst interval distributions (interburstintervaldistribution_in_seconds_N=X_beta=Y.csv)
- Definition: "X" represents the number of fireflies involved in each experiment, and Y represents the best fit beta value. X can be  5, 10, 15, or 20. 
- Columns:
    - Column 1 Interburst interval Tb [s]

- “Original” uses the envelope derived from the unedited single-firefly data.
- “Corrected” uses the envelope after trimming and removal.

----------------------------------------------------------------
###  Simulation_Code
This folder contains the code that was used to run the computational simulation.

1. Code structure:
Simulation.py and Firefly.py are objects used by run.py to instantiate and carry out the logic of the simulation. simulation_helpers.py contains a few helper functions for orchestration and extracting values from the data files mentioned below.

Note: inside each of the data folders are the corrected and originally uploaded files, in their respective folders. The code paths to these data files have been updated accordingly. 

#### Code Requirements (versions tested on):
- Python 3.7
- NumPy 1.22.1
- scipy 1.17.1
- pandas 1.3.3
- Matplotlib 3.4.3

The complete list of required packages provided in *requirements.txt*, which you can install in your environment with the command `pip install -r requirements.txt`. 

#### Usage:
Default parameters are already set up. To further set up or change parameters for the simulation, a number of flags can be set. Free parameters of the model discussed in the paper:
- `n`: Number of fireflies in arena (default: `20`)
- `beta`: Donation provided by any one flash (default: `0.2`)
- `no_refrac`: Run with this flag enabled via --no_refrac
- `use_linear`: Run with this flag enabled via --use_linear
- `one_flash`: Run with this flag enabled via --one_flash

For a list of customizable flags, please navigate to the Simulation_Code folder and run **`python run.py -h`**

**`python run.py`** runs a simulation with the default parameters.

#### Output:
In a newly created *simulation_results* within the larger Simulation_Results folder, there will be a pickle file named to match the density and beta parameters you provided. In here are dictionaries mapping each firefly to its respective a) flash time series (0s and 1s), b) voltage time series (continuous variable between 0 and 1), c) times of flashes (discrete values), d) positions at each itme step (discrete values), which can all be viewed by loading and unpickling the object.

Code for parsing the simulation pickles and exploring the heatmap space of the parameter sweep to find the best beta values is available upon request.
