"""
Synthesize Sequences (PAR)
In this notebook, we'll use the SDV library to create multiple, synthetic sequences. The SDV uses machine learning to learn patterns from real data and emulates them when creating synthetic data.
We'll use the PAR algorithm to do this. PAR uses a neural network to create sequences.
"""
import json

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

"""
# 1. Loading the demo data
"""

read_data_path = '../_fake_datasets/samples/nasdaq100_2019.csv'
metadata_path = '../_fake_datasets/samples/nasdaq100_2019_metadata.json'

# real_data, metadata = download_demo(
#     modality='sequential',
#     dataset_name='nasdaq100_2019'
# )
# real_data.to_csv(read_data_path, index=False, encoding='utf-8-sig')
#
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
metadata.sequence_key = loaded_metadata['sequence_key']
metadata.METADATA_SPEC_VERSION = loaded_metadata['METADATA_SPEC_VERSION']
metadata.sequence_index = loaded_metadata['sequence_index']
print(metadata)

# 1.1 What is sequential data?
# A sequence is a set of measurements taken in a particular order, such as the Open, Close and Volume of stock prices. Some datasets have a sequence index that prescribes this order. In our case, the Date column.
# In a single sequence, all measurements belong to the same entity. For example, if we isolate only the stock from Amazon (Symbol='AMZN'), then we have a single sequence of data. This sequence has 252 measurements with a Date ranging from the end of 2018 to 2019 .
amzn_sequence = real_data[real_data['Symbol'] == 'AMZN']
print(amzn_sequence)
print(real_data['Symbol'].unique())

# 1.2 What are Context Columns?
# A context column does not change during the course of a sequence. In our case, Sector and Industry are context columns.
# If we choose a sequence -- such as Amazon (Symbol='AMZN') -- then we'll see that the context values don't change. Amazon is always a 'Consumer Services' company.
print(real_data[real_data['Symbol'] == 'AMZN']['Sector'].unique())

"""
# 2. Basic Usage
"""

# 2.1 Creating a Synthesizer
# An SDV synthesizer is an object that you can use to create synthetic data. It learns patterns from the real data and replicates them to generate synthetic data.
synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Sector', 'Industry'],
    verbose=True)

synthesizer.fit(real_data)

# 2.2 Generating Synthetic Data
# Use the sample function and pass in any number of sequences to synthesize. The synthesizer algorithmically determines how long to make each sequence.
synthetic_data = synthesizer.sample(num_sequences=10)
print(synthetic_data.head())

print(synthetic_data[['Symbol', 'Industry']].groupby(['Symbol']).first().reset_index())

# 2.3 Saving and Loading
# We can save the synthesizer to share with others and sample more synthetic data in the future.
synthesizer.save('./results/par_synthesizer.pkl')
synthesizer = PARSynthesizer.load('../results/_backup/par_synthesizer.pkl')

"""
# 3. PAR Customization
"""
# We can customizer our PARSynthesizer in many ways.
# - Use the `epochs` parameter to make a tradeoff between training time and data quality. Higher epochs mean the synthesizer will train for longer, ideally improving the data quality.
# - Use the `enforce_min_max_values` parameter to specify whether the synthesized data should always be within the same min/max ranges as the real data. Toggle this to `False` in order to enable forecasting.
custom_synthesizer = PARSynthesizer(
    metadata,
    epochs=250,
    context_columns=['Sector', 'Industry'],
    enforce_min_max_values=False,
    verbose=True)

custom_synthesizer.fit(real_data)

"""
# 4. Sampling Options
"""
# Using the PAR synthesizer, you can customize the synthetic data to suit your needs.

# 4.1 Specify Sequence Length
# By default, the synthesizer algorithmically determines the length of each sequence. However, you can also specify a fixed, predetermined length.
results = custom_synthesizer.sample(num_sequences=3, sequence_length=2)
print(results)

long_sequence = custom_synthesizer.sample(num_sequences=1, sequence_length=500)
print(long_sequence.tail())

# 4.2 Conditional Sampling Using Context
# You can pass in context columns and allow the PAR synthesizer to simulate the sequence based on those values.
# Let's start by creating a scenario with 2 companies in the Technology sector and 3 others in the Consumer Services sector. Each row corresponds to a new sequence that we want to synthesize.
scenario_context = pd.DataFrame(data={
    'Symbol': ['COMPANY-A', 'COMPANY-B', 'COMPANY-C', 'COMPANY-D', 'COMPANY-E'],
    'Sector': ['Technology'] * 2 + ['Consumer Services'] * 3,
    'Industry': ['Computer Manufacturing', 'Computer Software: Prepackaged Software',
                 'Hotels/Resorts', 'Restaurants', 'Clothing/Shoe/Accessory Stores']
})
print(scenario_context)

results = custom_synthesizer.sample_sequential_columns(
    context_columns=scenario_context,
    sequence_length=2
)
print(results)
