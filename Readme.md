
# Neighbors' Influence on Predicting Focal Fish's Velocity

## Overview
In order to find answers for our research question the respitory was created.

RQ1: How well can we predict fish behavior based on its neighbors' cues?

RQ2: Which neighbors provide the most accurate prediction of the fish's velocity?

RQ3: What kind of information regarding the neighbor information is the basis for this choice?

To answer the research questions:
- developed an evolutionary algorithm to optimize prediction accuracy.
- visualized weight distributions to interpret each group member's influence.
- simulated behavior cues (bearing, distance, orientation)
- tested if the wall has a signifiant impact on predicting the focal fish's velocity.
- compared predicted and actual agent movements to evaluate model performance.

## Datasets
We make use of datasets which have been kindly and generously published by others. The data files are labeled with references to the papers which have originally produced and published them. We do not claim any rights or responsibility for these datasets.

Lei, L., Escobedo, R., Sire, C., & Theraulaz, G. (2020). Computational and robotic modeling reveal parsimonious combinations of interactions between individuals in schooling fish. PLoS computational biology, 16(3), e1007194. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007194

## How to use
Before any experiments can be run, you need to make sure that the augmented data is present in the data/ subfolder. If that is not the case, please first run data_preparation.py which will augment the data ready for use in the EA experiments.

Once this is ensured, create a config under configs/ to suit your experimental needs. Then create a run script and run.

## Testing
This implementation comes with a set of tests which can be run by running 'pytest' or 'python -m pytest' in the terminal.

## Libraries
-Python (3.12.3)
-NumPy
-Pandas
-Matplotlib
-SciPy
-Scikit-learn
-Seaborn
-PyTest