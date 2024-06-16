"""
Synthesize a Table (Gaussian Coupla)

In this notebook, we'll use the SDV to create synthetic data for a single table and evaluate it. The SDV uses machine learning to learn patterns from real data and emulates them when creating synthetic data.

We'll use the **Gaussian Copula** algorithm to do this. Gaussian Copula is a fast, customizable and transparent way to synthesize data.

_Last Edit: Dec 12, 2023_
"""

import json

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_pair_plot
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer

"""
# 1. Loading the demo data
"""

# For this demo, we'll use a fake dataset that describes some fictional guests staying at a hotel.
read_data_path = '../datasets/fake_hotel_guests.csv'
metadata_path = './datasets/metadata.json'

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
#

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

"""
# 2. Basic Usage
"""

# 2.1 Creating a Synthesizer
# An SDV synthesizer is an object that you can use to create synthetic data. It learns patterns from the real data and replicates them to generate synthetic data.
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

# 2.2 Generating Synthetic Data
# Use the sample function and pass in any number of rows to synthesize.
synthetic_data = synthesizer.sample(num_rows=500)
synthetic_data.head()

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

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata
)

quality_report.get_details('Column Shapes')

# 2.4 Visualizing the Data
# For more insights, we can visualize the real vs. synthetic data.
# Let's perform a 1D visualization comparing a column of the real data to the synthetic data.
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='room_rate',
    metadata=metadata
)
fig.show()

fig = get_column_pair_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_names=['room_rate', 'room_type'],
    metadata=metadata
)
fig.show()

# 2.5 Anonymization
# In the original dataset, we had some sensitive columns such as the guest's email, billing address and phone number. In the synthetic data, these columns are fully anonymized -- they contain entirely fake values that follow the format of the original.
# PII columns are not included in the quality report, but we can inspect them to see that they are different.
sensitive_column_names = ['guest_email', 'billing_address', 'credit_card_number']
real_data[sensitive_column_names].head(3)
print(real_data[sensitive_column_names].head(3))
print(synthetic_data[sensitive_column_names].head(3))

# 2.6 Saving and Loading
# We can save the synthesizer to share with others and sample more synthetic data in the future.
synthesizer.save('./results/ga_copula_synthesizer.pkl')
synthesizer = GaussianCopulaSynthesizer.load('../results/_backup/ga_copula_synthesizer.pkl')

"""
# 3. Gaussian Copula Customization
"""
# A key benefit of using the Gaussian Copula is customization and transparency. This synthesizer estimates the shape of every column using a 1D distribution. We can set these shapes ourselves.
custom_synthesizer = GaussianCopulaSynthesizer(
    metadata,
    default_distribution='truncnorm',
    numerical_distributions={
        'checkin_date': 'uniform',
        'checkout_date': 'uniform',
        'room_rate': 'gaussian_kde'
    }
)
custom_synthesizer.fit(real_data)

learned_distributions = custom_synthesizer.get_learned_distributions()
print(learned_distributions['has_rewards'])

synthetic_data_customized = custom_synthesizer.sample(num_rows=500)
quality_report = evaluate_quality(
    real_data,
    synthetic_data_customized,
    metadata
)

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data_customized,
    column_name='room_rate',
    metadata=metadata
)
fig.show()
fig.write_image('./results/ca_copula_1d_col.png')

"""
# 4. Conditional Sampling
"""
# Another benefit of using the Gaussian Copula is the ability to efficiently sample conditions. This allows us to simulate hypothetical scenarios.
# Let's start by creating a scenario where every hotel guest is staying in a SUITE (half with rewards and half without).
suite_guests_with_rewards = Condition(
    num_rows=250,
    column_values={'room_type': 'SUITE', 'has_rewards': True}
)

suite_guests_without_rewards = Condition(
    num_rows=250,
    column_values={'room_type': 'SUITE', 'has_rewards': False}
)

simulated_synthetic_data = custom_synthesizer.sample_from_conditions(conditions=[
    suite_guests_with_rewards,
    suite_guests_without_rewards
])

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=simulated_synthetic_data,
    column_name='room_type',
    metadata=metadata
)

fig.update_layout(
    title='Using synthetic data to simulate room_type scenario'
)
fig.show()
fig.write_image('./results/ca_copula_room_type_scenario.png')
