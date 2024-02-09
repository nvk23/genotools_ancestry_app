import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from hold_data import config_page, data_count, meta_ancestry_select

config_page('Metadata')
data_count()

# gets master key (full GP2 release or selected cohort)
master_key = st.session_state['master_key']

st.title(f'Metadata')

# remove pruned samples
master_key = master_key[master_key['pruned'] == 0]

# metadata ancestry selection
meta_ancestry_select(master_key)
meta_ancestry_choice = st.session_state['meta_ancestry_choice']

if meta_ancestry_choice != 'All':
    master_key = master_key[master_key['label'] == meta_ancestry_choice]

plot1, plot2 = st.columns([1,1])

master_key.rename(columns = {'sex_for_qc': 'Sex', 'phenotype':'Phenotype'}, inplace = True)

master_key['Sex'].replace(1, 'Male', inplace = True)
master_key['Sex'].replace(2, 'Female', inplace = True)
master_key['Sex'].replace(0, 'Unknown', inplace = True)
master_key['Sex'].replace(-9, 'Unknown', inplace = True)

master_key['Phenotype'].replace(1, 'Control', inplace = True)
master_key['Phenotype'].replace(2, 'Case', inplace = True)
master_key['Phenotype'].replace(0, 'Unknown', inplace = True)
master_key['Phenotype'].replace(-9, 'Unknown', inplace = True)

male_pheno = master_key.loc[master_key['Sex'] == 'Male', 'Phenotype']
female_pheno = master_key.loc[master_key['Sex'] == 'Female', 'Phenotype']

combined_counts = pd.DataFrame()
combined_counts['Male'] = male_pheno.value_counts()
combined_counts['Female'] = female_pheno.value_counts()
combined_counts = combined_counts.transpose()
combined_counts['Total'] = combined_counts.sum(axis=1)
combined_counts = combined_counts.fillna(0)
combined_counts = combined_counts.astype('int32')

plot1.markdown('#### Phenotype Count Split by Sex')
plot1.dataframe(combined_counts, use_container_width=True)
