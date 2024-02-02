import os
import sys
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
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