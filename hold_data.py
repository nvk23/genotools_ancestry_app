import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import json
from io import StringIO

from google.cloud import storage

# functions used on every pages

# config page with gp2 logo in browser tab
def config_page(title):
    if 'gp2_bg' in st.session_state:
        st.set_page_config(
            page_title=title,
            page_icon=st.session_state.gp2_bg,
            layout="wide",
        )
    else:
        gp2_bg = 'logos/gp2_2.jpg'
        st.session_state['gp2_bg'] = 'logos/gp2_2.jpg'
        st.set_page_config(
            page_title=title,
            page_icon=gp2_bg,
            layout="wide"
        )

# load and place sidebar logos
def place_logos():
    sidebar1, sidebar2 = st.sidebar.columns(2)
    if ('card_removebg' in st.session_state):
        sidebar1.image(st.session_state.card_removebg, use_column_width=True)
        sidebar2.image(st.session_state.gp2_removebg, use_column_width=True)
    else:
        card_removebg = 'logos/card-removebg.png'
        gp2_removebg = 'logos/gp2_2-removebg.png'
        st.session_state['card_removebg'] = card_removebg
        st.session_state['gp2_removebg'] = gp2_removebg
        sidebar1.image(card_removebg, use_column_width=True)
        sidebar2.image(gp2_removebg, use_column_width=True)

# Sidebar selectors
def meta_ancestry_callback():
    st.session_state['old_meta_ancestry_choice'] = st.session_state['meta_ancestry_choice']
    st.session_state['meta_ancestry_choice'] = st.session_state['new_meta_ancestry_choice']

def meta_ancestry_select(master_key):
    st.markdown('### **Choose an ancestry!**')

    meta_ancestry_options = ['All'] + [label for label in master_key['label'].dropna().unique()]

    if 'meta_ancestry_choice' not in st.session_state:
        st.session_state['meta_ancestry_choice'] = meta_ancestry_options[0]

    if st.session_state['meta_ancestry_choice'] not in meta_ancestry_options:
        st.error(f"No samples with {st.session_state['meta_ancestry_choice']} ancestry in {st.session_state['cohort_choice']}. \
                 Displaying all ancestries instead!")
        st.session_state['meta_ancestry_choice'] = meta_ancestry_options[0]

    if 'old_meta_ancestry_choice' not in st.session_state:
        st.session_state['old_chr_choice'] = ""
    
    st.session_state['meta_ancestry_choice'] = st.selectbox(label='Ancestry Selection', label_visibility = 'collapsed', options=meta_ancestry_options, index=meta_ancestry_options.index(st.session_state['meta_ancestry_choice']), key='new_meta_ancestry_choice', on_change=meta_ancestry_callback)

def data_count():
    total_count = st.session_state['master_key'].shape[0]
    pruned_samples = sum(st.session_state['master_key'].pruned)

    st.sidebar.metric("", 'Overview of Your Data')
    st.sidebar.metric("Number of Samples in Dataset:", f'{total_count:,}')
    st.sidebar.metric("Number of Samples After Pruning:", f'{(total_count-pruned_samples):,}')

    # place logos in sidebar
    st.sidebar.markdown('---')

    place_logos()


# Derive master key from GenoTools input/output paths
def create_master_key(geno_path, out_path):
    ancestry_labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
    related_list = ['first_deg', 'second_deg']
    out_dir = os.path.dirname(os.path.abspath(out_path))

    # load in psam file (input to genotools)
    # check if inputs in PLINK 1 or 2 before this - if plink1 run bfiles to pfiles
    psam = pd.read_csv(f'{geno_path}.psam', sep='\s+')
    psam = psam.drop(columns=['PAT','MAT'], axis=1)
    psam = psam.rename({'#FID':'FID','PHENO1':'PHENO'}, axis=1)

    # load in JSON (output from genotools)
    f = open(f'{out_path}.json')
    data = json.load(f)

    # read json file and extract parts
    qc_df = pd.DataFrame(data['QC'])
    al = pd.DataFrame(data['ancestry_labels'])
    ps = pd.DataFrame(data['pruned_samples'])
    ppcs = pd.DataFrame(data['projected_pcs'])
    rpcs = pd.DataFrame(data['ref_pcs'])
    cm = pd.DataFrame(data['confusion_matrix'])

    # extras from JSON - don't need
    # ac = pd.DataFrame(data['ancestry_counts'])
    # numap = pd.DataFrame(data['new_samples_umap'])
    # rumap = pd.DataFrame(data['ref_umap'])
    # tumap = pd.DataFrame(data['total_umap'])

    # begin creating final key
    final_key = al.merge(psam, on=['FID','IID'], how='left')

    # track pruning
    final_key['pruned'] = np.where(final_key['IID'].isin(ps.IID), 1, 0)
    final_key = final_key.merge(ps[['IID','step']], how='left', on=['IID'])
    final_key['step'] = np.where(final_key['step'] == 'sex', 'sex_prune', final_key['step'])
    final_key = final_key.rename({'step':'pruned_reason'}, axis=1)

    # track relatedness - create list with all .related files in out_path directory
    related_files = pd.DataFrame()
    for file in os.listdir(out_dir):
        if file.endswith(".related"):
            related_path = os.path.join(out_dir, file)
            df = pd.read_csv(related_path)
            related_files = pd.concat([related_files, df])

    related = related_files[related_files['REL'].isin(related_list)]
    related_files_list = set(list(related['IID1']) + list(related['IID2']))
    related_samples = final_key[['IID']]
    related_samples['related'] = np.where(related_samples['IID'].isin(related_files_list), 1, 0)
    final_key = final_key.merge(related_samples, on=['IID'], how='left')

    # finalize and save derived master key
    final_key = final_key.rename({'SEX':'sex_for_qc','AGE':'age','PHENO':'phenotype'}, axis=1)
    final_key['sex_for_qc'].fillna(0, inplace = True)

    # create session states
    if 'master_key' not in st.session_state:
        st.session_state['master_key'] = final_key
    if 'qc_df' not in st.session_state:
        st.session_state['qc_df'] = qc_df
    if 'ref_pcs' not in st.session_state:
        st.session_state['ref_pcs'] = rpcs
    if 'proj_pcs' not in st.session_state:
        st.session_state['proj_pcs'] = ppcs
    if 'confusion_matrix' not in st.session_state:
        st.session_state['confusion_matrix'] = cm