import pandas as pd

from syntheval import SynthEval

### In this frame we load the dataframes and define some characteristics of them that we will use later.

target_column = 'DPS_AC_SN'                           # column to use as target for classification metrics and coloration of PCA plot.
categorical_columns = ['species','island','sex']    # Categorical columns can be either supplied or automatically inferred using a number of unique values threshold.

df_real = pd.read_csv('../datasets/hf_sample_1000.csv')
df_fake = pd.read_csv('../datasets/hf_sample_10000.csv')

### Testing data is not required, but the usability analysis will be more complete if it is included.
# df_test = pd.read_csv(load_dir + filename + '_test.csv')

### First SynthEval object is created then run with the "full_eval" presets file.
S = SynthEval(df_real)
results = S.evaluate(df_fake,target_column,"full_eval")   # The _ is for Jupyter purposes only, to avoid printing the results dictionary as well

print(results)
