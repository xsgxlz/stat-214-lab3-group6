### Exploratory Data Analysis

##### Contents
This directory contains two primary scripts:

1. `eda_cleaning.ipynb`: This Python notebook processes the three raw `.npy` files containing the fMRI data for both subjects. It loads the data, computes various summary statistics at different levels of aggregation, and saves each dataset as a `.csv` file in the `data` directory.
   
2. `eda_plots.ipynb`: This R notebook loads the cleaned `.csv` files and generates various exploratory data analysis plots. The resulting figures are saved to the `figs` directory.

##### Relevant Data

- `clean.csv`: each row contains fMRI signal summary statistics (for both subject 2 and 3) at a single time point within a single story.
- `overall.csv`: each row contains fMRI signal summary statistics (for both subject 2 and 3) for an entire story's data (i.e., across all time points).
- `words.csv`: counts the number of words and characters within the plaintext associated with each story.

##### Dependencies

- Python: 
  - `numpy`
  - `pandas`
  - `os`
  - `os`
  - `pickle`
  - `matplotlib`
  - (See `environment.yaml` for full details)
 
- R: 
  - `tidyverse`
  - (See `environment-r.yaml` for full details)
