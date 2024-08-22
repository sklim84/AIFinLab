"""
Synthesize a Table (CTGAN)
In this notebook, we'll use the SDV to create synthetic data for a single table and evaluate it. The SDV uses machine learning to learn patterns from real data and emulates them when creating synthetic data.

We'll use the CTGAN algorithm to do this. CTGAN uses generative adversarial networks (GANs) to create synthesize data with high fidelity.

Last Edit: Dec 12, 2023
"""

import json

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_pair_plot
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

"""
# 1. Loading the demo data
"""

# For this demo, we'll use a fake dataset that describes some fictional guests staying at a hotel.
read_data_path = '../_fake_datasets/samples/fake_hotel_guests.csv'
metadata_path = '../_fake_datasets/samples/fake_hotel_guests_metadata.json'

# real_data, metadata = download_demo(
#     modality='single_table',
#     dataset_name='fake_hotel_guests'
# )
# real_data.to_csv(read_data_path, index=False, encoding='utf-8-sig')
# print(metadata)
# metadata = metadata.to_dict()
# print(metadata)
# with open(metadata_path, 'w') as json_file:
#     json.dump(metadata, json_file, indent=4)

# Details: The data is available as a single table.
# guest_email is a primary key that uniquely identifies every row
# Other columns have a variety of data types and some the data may be missing.
real_data = pd.read_csv(read_data_path)
print(real_data.head())

with open(metadata_path, 'r') as json_file:
    loaded_metadata = json.load(json_file)
metadata = SingleTableMetadata()
metadata.columns = loaded_metadata['columns']
metadata.primary_key = loaded_metadata['primary_key']
metadata.METADATA_SPEC_VERSION = loaded_metadata['METADATA_SPEC_VERSION']
print(metadata)

# The demo also includes metadata, a description of the dataset. It includes the primary keys as well as the data types for each column (called "sdtypes").
# metadata.visualize(output_filepath='./results/ctgan_metadata.png')

"""
# 2. Basic Usage
"""

# 2.1 Creating a Synthesizer
# An SDV synthesizer is an object that you can use to create synthetic data. It learns patterns from the real data and replicates them to generate synthetic data.
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(real_data)

# 2.2 Generating Synthetic Data
# Use the sample function and pass in any number of rows to synthesize.
synthetic_data = synthesizer.sample(num_rows=500)
print(synthetic_data.head())

# 2.3 Evaluating Real vs. Synthetic Data
# SDV has built-in functions for evaluating the synthetic data and getting more insight.
# As a first step, we can run a diagnostic to ensure that the data is valid. SDV's diagnostic performs some basic checks such as:
# All primary keys must be unique
# Continuous values must adhere to the min/max of the real data
# Discrete columns (non-PII) must have the same categories as the real data
# Etc.
diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# We can also measure the data quality or the statistical similarity between the real and synthetic data. This value may vary anywhere from 0 to 100%.\
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

# According to the score, the synthetic data is about 75% similar to the real data in terms of statistical similarity.
# We can also get more details from the report. For example, the Column Shapes sub-score is 76%. Which columns had the highest vs. the lowest scores?
quality_report.get_details('Column Shapes')

# 2.4 Visualizing the Data
# For more insights, we can visualize the real vs. synthetic data.
# Let's perform a 1D visualization comparing a column of the real data to the synthetic data.
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='room_type',
    metadata=metadata
)
fig.show()
fig.write_image('./results/ctgan_1d_col.png')

# We can also visualize in 2D, comparing the correlations of a pair of columns.
fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['room_rate', 'room_type'],
    metadata=metadata
)
fig.show()
fig.write_image('./results/ctgan_2d_corr.png')

# 2.5 Anonymization
# In the original dataset, we had some sensitive columns such as the guest's email, billing address and phone number. In the synthetic data, these columns are fully anonymized -- they contain entirely fake values that follow the format of the original.
# PII columns are not included in the quality report, but we can inspect them to see that they are different.
sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']
print(real_data[sensitive_column_names].head(3))
print(synthetic_data[sensitive_column_names].head(3))

# 2.6 Saving and Loading
# We can save the synthesizer to share with others and sample more synthetic data in the future.
synthesizer.save('./results/ctgan_synthesizer.pkl')
synthesizer = CTGANSynthesizer.load('../results/_backup/ctgan_synthesizer.pkl')

"""
# 3. CTGAN Customization
"""
# When using this synthesizer, we can make a tradeoff between training time and data quality using the epochs parameter: Higher epochs means that the synthesizer will train for longer, and ideally improve the data quality.
custom_synthesizer = CTGANSynthesizer(
    metadata,
    epochs=1000)
custom_synthesizer.fit(real_data)

# After we've trained our synthesizer, we can verify the changes to the data quality by creating some synthetic data and evaluating it.
synthetic_data_customized = custom_synthesizer.sample(num_rows=500)
quality_report = evaluate_quality(
    real_data,
    synthetic_data_customized,
    metadata
)

# While GANs are able to model complex patterns and shapes, it is not easy to understand how they are learning -- but it is possible to modify the underlying architecture of the neural networks.
# For users who are familiar with the GAN architecture, there are extra parameters you can use to tune CTGAN to your particular needs. For more details, see the CTGAN documentation.
