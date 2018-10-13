# HA_stalkbnAb_MAP

Mutational antigenic profiling of stalk bnAbs in H1 vs H3.

This project is in collaboration with Nicholas Wu. 
Experiments and computational analysis performed in this repository were performed by Juhye Lee (Bloom Lab).

Briefly, the analysis performed here maps escape mutations in the WSN/1933 H1 HA and the Perth/2009 H3 HA from two broadly-neutralizing stalk-binding antibodies, CR9114 and FI6v3.
Escape of the WSN/1933 H1 from FI6v3 is from data in [Doud, Lee, Bloom (2018)](https://www.nature.com/articles/s41467-018-03665-3).

## Organization

This repository is organized into the following subdirectories:

  * [./data/](./data/): contains all required input for the analyses.
  * [./results/](./results/): contains all generated results from the analyses

#### Subdirectory Organization

The [./data/](./data/) directory contains the following items:
  * [./data/H1toH3_renumber.csv](./data/H1toH3_renumber.csv) is a CSV file that allows for conversion from sequential numbering of the WSN H1 to H3 numbering. The sequential numbering is in the _original_ column and the H3 numbering is in the _new_ column.
  * [./data/H3renumbering_scheme.csv](./data/H3renumbering_scheme.csv) is a CSV file that allows for conversion from sequential numbering of the Perth/2009 H3 to H3 numbering. This file is formatted similarly to the `H1toH3_renumber` file described above.
  * [./data/Perth09_HA_reference.fa](./data/Perth09_HA_reference.fa) contains the wildtype sequence of the A/Perth/16/2009 (H3N2) HA used for these experiments.
  * [./data/WSN_HA_reference.fa](./data/WSN_HA_reference.fa) contains the wildtype sequence of the A/WSN/1933 (H1N1) HA used for these experiments.
  * [./data/samples.csv](./data/samples.csv) is a CSV that lists the samples, the SRA run that contains the deep sequencing data for each sample, and other sample metadata.

The following subdirectories in the [./results/](./results/) directory are included in this repository and _not_ included in the `.gitignore`:
  * [./results/renumberedcounts/](./results/renumberedcounts/): contains codon counts of each sample in H3 numbering as a `csv` file.
  * [./results/fracsurviveaboveavg/](./results/fracsurviveaboveavg/): contains the fraction survive values above the overall library average, in `csv` files. Also contains the logo plots for the entire gene for each sample and the averaged samples.
  * [./results/plots/](./results/plots/): contains the zoomed logo plots in `pdf` format.

The analysis is performed by the [analysis_notebook.ipynb](analysis_notebook.ipynb) Jupyter notebook.
The analysis is performed using the [dms_tools2](https://jbloomlab.github.io/dms_tools2/) software.

