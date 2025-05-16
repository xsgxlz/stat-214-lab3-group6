### Interpretation (Lab 3.3 Part 2)

##### Contents
This directory contains three primary notebooks:

1. `run_shap_lime.ipynb`: This Python notebook loads the BERT models stored on the PSC and then runs SHAP and LIME, saving the results to `{score}_{story}_subj{subject}.pkl` (8 separate files, not on GitHub due to file sizes). Code is heavily barrowed from `pretrained_bert_ridge_extract.ipynb` to extract predictions from the model.

2. `clean.ipynb`: This Python notebook loads the `.pkl` files from above, flattens them from 3D to 2D, cleans them, then saves them to `{score}_{story}_subj{subject}.csv` (8 separate files, found in `data` directory).

3. `importance_plots.ipynb` : This R notebook loads the `.csv` files from above, merges and cleans the data further, then saves it to `{story}.csv` (2 separate files, found in `data` directory). The remainder of the notebook generates various plots for analysis of SHAP and LIME values (found in `figs` directory).
4. `lab 3 code.qmd`: This python qmd file loads the `.pkl` files from above and creates several scatterplots, barplots and distribution plots to analyse the relationship and differences between SHAP and LIME methods


##### Relevant Data

- `{score}_{story}_subj{subject}.pkl`: output of `run_shap_lime.ipynb`, input to `clean.ipynb` (8 separate files, not on GitHub due to file sizes)
- `{score}_{story}_subj{subject}.csv`: output of `clean.ipynb`, input to `importance_plots.ipynb` (8 separate files, found in `data` directory)
- `{story}.csv`: output of `importance_plots.ipynb` (2 separate files, found in `data` directory), one of various dataframe formats used to make plots.

##### Dependencies

- Python: 
  - `numpy`
  - `seaborn`
  - `scipy.stats`
  - `spearmanr`
  - `pandas`
  - `os`
  - `pickle`
  - `time`
  - `random`
  - `copy`
  - `itertools`
  - `matplotlib`
  - `sklearn`
  - `BERT` (local)
  - `ridge_utils` (local)
  - `torch`
  - `transformers`
  - `peft`
  - `scipy`
  - `shap`
  - `lime`
  - (See `environment.yaml` for full details)

 
- R: 
  - `tidyverse`
  - `rlang`
  - (See `environment-r.yaml` for full details)
