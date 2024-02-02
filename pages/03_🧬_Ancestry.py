import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from hold_data import config_page, place_logos


# Plots 3D PCA
def plot_3d(labeled_df, color, symbol=None, x='PC1', y='PC2', z='PC3', title=None, x_range=None, y_range=None, z_range=None):
    '''
    Parameters: 
    labeled_df (Pandas dataframe): labeled ancestry dataframe
    color (string): color of ancestry label. column name containing labels for ancestry in labeled_pcs_df
    symbol (string): symbol of secondary label (for example, predicted vs reference ancestry). default: None
    x (string): column name of x-dimension
    y (string): column name of y-dimension
    z (string): column name of z-dimension
    title (string, optional): title of output scatterplot
    x_range (list of floats [min, max], optional): range for x-axis
    y_range (list of floats [min, max], optional): range for y-axis
    z_range (list of floats [min, max], optional): range for z-axis

    Returns:
    3-D scatterplot (plotly.express.scatter_3d). If plot_out included, will write .png static image and .html interactive to plot_out filename
                
    '''

    fig = px.scatter_3d(
                labeled_df,
                x=x,
                y=y,
                z=z,
                color=color,
                symbol=symbol,
                title=title,
                color_discrete_sequence=px.colors.qualitative.Bold,
                range_x=x_range,
                range_y=y_range,
                range_z=z_range,
                hover_name="IID",
                color_discrete_map={'AFR': "#88CCEE",
                                    'SAS': "#CC6677",
                                    'EAS': "#DDCC77",
                                    'EUR':"#117733",
                                    'AMR':"#332288",
                                    'AJ': "#D55E00",
                                    'AAC':"#999933",
                                    'CAS':"#882255",
                                    'MDE':"#661100",
                                    'FIN':"#F0E442",
                                    'CAH':"#40B0A6",
                                    'Predicted':"#ababab"}
            )

    fig.update_traces(marker={'size': 3})
    st.plotly_chart(fig)

# Plots pie chart
def plot_pie(df):
    '''
    Parameters:
    df (Pandas dataframe): dataframe with ancestry categories and respective counts
    
    Returns:
    interactive pie chart (plotly.express.pie)
    '''
    pie_chart = px.pie(df, names = 'Ancestry Category', values = 'Proportion', hover_name = 'Ancestry Category',  color="Ancestry Category", 
                    color_discrete_map={'AFR': "#88CCEE",
                                        'SAS': "#CC6677",
                                        'EAS': "#DDCC77",
                                        'EUR':"#117733",
                                        'AMR':"#332288",
                                        'AJ': "#D55E00",
                                        'AAC':"#999933",
                                        'CAS':"#882255",
                                        'MDE':"#661100",
                                        'FIN':"#F0E442",
                                        'CAH':"#40B0A6"})
    pie_chart.update_layout(showlegend = True, width=500,height=500)
    st.plotly_chart(pie_chart)


config_page('Ancestry')

place_logos()

# Gets master key (full GP2 release or selected cohort)
master_key_path = f'data/master_key_release6_final.csv'
master_key = pd.read_csv(master_key_path, sep=',')

# remove pruned samples
master_key = master_key[master_key['pruned'] == 0]

# Tab navigator for different parts of Ancestry Method
tabPCA, tabPredStats, tabPie, tabMethods = st.tabs(["Ancestry Prediction", "Model Performance", "Ancestry Distribution","Method Description"])

with tabPCA:
    ref_pca = pd.read_csv(f'data/reference_pcs.csv', sep=',')
    proj_pca = pd.read_csv(f'data/projected_pcs.csv', sep=',')
    proj_labels = pd.read_csv(f'data/pred_labels.txt', sep='\s+')

    proj_pca = proj_pca.drop(columns=['label'], axis=1)

    # Projected PCAs will have label "Predicted", Reference panel is labeled by ancestry 
    proj_pca = proj_pca.merge(proj_labels[['FID','IID','label']], how='inner', on=['FID','IID'])
    proj_pca['plot_label'] = 'Predicted'
    ref_pca['plot_label'] = ref_pca['label']

    total_pca = pd.concat([ref_pca, proj_pca], axis=0)
    new_labels = proj_pca['label']

    pca_col1, pca_col2 = st.columns([1.5,3])
    st.markdown('---')
    col1, col2 = st.columns([1.5, 3])

    # Get actual ancestry labels of each sample in Projected PCAs instead of "Predicted" for all samples
    combined = proj_pca[['IID', 'label']]
    combined_labelled = combined.rename(columns={'label': 'Predicted Ancestry'})
    holdValues = combined['label'].value_counts().rename_axis('Predicted Ancestry').reset_index(name='Counts')

    with pca_col1:
        st.markdown(f'### Reference Panel vs. UKBB Sample PCA')
        with st.expander("Description"):
            st.write('Select an Ancestry Category below to display only the Predicted samples within that label.')

        holdValues['Select'] = False
        select_ancestry = st.data_editor(holdValues, hide_index=True, use_container_width=True)
        selectionList = select_ancestry.loc[select_ancestry['Select'] == True]['Predicted Ancestry']

    with pca_col2:
        # If category selected, plots Projected PCA samples in that category under label "Predicted"
        if not selectionList.empty:
            selected_pca = proj_pca.copy()
            selected_pca.drop(selected_pca[np.logical_not(selected_pca['label'].isin(selectionList))].index, inplace = True)
            
            for items in selectionList:  # subsets Projected PCA by selected categories
                selected_pca.replace({items: 'Predicted'}, inplace = True)

            total_pca_selected = pd.concat([ref_pca, selected_pca], axis=0)
            plot_3d(total_pca_selected, 'label')
        else:
            plot_3d(total_pca, 'plot_label')  # if no category selected, plots all samples of Projected PCA with "Predicted" label

    with col1:
        st.markdown(f'### UKBB Sample PCA')
        with st.expander("Description"):
            st.write('All Predicted samples and their respective labels are listed below. Click on the table and use ⌘ Cmd + F or Ctrl + F to search for specific samples.')
        # combined_labelled = combined_labelled.set_index('IID')
        st.dataframe(combined_labelled, hide_index=True, use_container_width=True)
        
    with col2: 
        plot_3d(proj_pca, 'label')  # only plots PCA of predicted samples


with tabPredStats:
    st.markdown(f'## **Model Accuracy**')
    confusion_matrix = pd.read_csv('data/confusion_matrix.csv', sep=',')

    if 'label' in confusion_matrix.columns:
        confusion_matrix.set_index('label', inplace=True)
    elif 'Unnamed: 0' in confusion_matrix.columns:
        confusion_matrix = confusion_matrix.rename({'Unnamed: 0':'label'}, axis=1)
        confusion_matrix.set_index('label', inplace = True)
    else:
        confusion_matrix.set_index(confusion_matrix.columns, inplace = True)

    tp = np.diag(confusion_matrix)
    col_sum = confusion_matrix.sum(axis=0)
    row_sum = confusion_matrix.sum(axis=1)

    class_recall = np.array(tp/row_sum)
    class_precision = np.array(tp/col_sum)

    balanced_accuracy = np.mean(class_recall)
    margin_of_error = 1.96 * np.sqrt((balanced_accuracy*(1-balanced_accuracy))/sum(col_sum))

    precision = np.mean(class_precision)

    f1 = np.mean(2 * ((class_recall * class_precision)/(class_recall + class_precision)))

    heatmap1, heatmap2 = st.columns([2, 1])

    with heatmap1:
        st.markdown('### Confusion Matrix')
        fig = px.imshow(confusion_matrix, labels=dict(x="Predicted Ancestry", y="Reference Panel Ancestry", color="Count"), text_auto=True, color_continuous_scale='plasma')
        st.plotly_chart(fig)

    # Plots heatmap of confusion matrix from Testing
    with heatmap2:
        st.markdown('### Test Set Performance')
        st.markdown('#')
        st.metric('Balanced Accuracy:', "{:.3f} \U000000B1 {:.3f}".format(round(balanced_accuracy, 3), round(margin_of_error, 3)))
        st.markdown('#')
        st.metric('Precision:', "{:.3f}".format(round(precision, 3)))
        st.markdown('#')
        st.metric('F1 Score:', "{:.3f}".format(round(f1, 3)))

with tabPie:
    # Plots ancestry breakdowns of Predicted Samples vs. Reference Panel samples
    pie1, pie2, pie3 = st.columns([2,1,2])
    p1, p2, p3 = st.columns([2,4,2])

    ref_pca = pd.read_csv('data/reference_pcs.csv', sep=',')

    # Get dataframe of counts per ancestry category for reference panel
    df_ancestry_counts = ref_pca['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
    ref_counts = ref_pca['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
    ref_combo = pd.merge(df_ancestry_counts, ref_counts, on='Ancestry Category')
    ref_combo.rename(columns = {'Proportion': 'Ref Panel Proportion', 'Counts': 'Ref Panel Counts'}, inplace = True)

    # Gets dataframe of counts per ancestry category for predicted samples
    df_new_counts = master_key['label'].value_counts(normalize = True).rename_axis('Ancestry Category').reset_index(name='Proportion')
    new_counts = master_key['label'].value_counts().rename_axis('Ancestry Category').reset_index(name='Counts')
    new_combo = pd.merge(df_new_counts, new_counts, on='Ancestry Category')
    new_combo.rename(columns = {'Proportion': 'Predicted Proportion', 'Counts': 'Predicted Counts'}, inplace = True)

    ref_combo = ref_combo[['Ancestry Category', 'Ref Panel Counts']]
    ref_combo_cah = pd.DataFrame([['CAH', 'NA']], columns=['Ancestry Category', 'Ref Panel Counts'])
    ref_combo = pd.concat([ref_combo, ref_combo_cah], axis=0)

    pie_table = pd.merge(ref_combo, new_combo, on='Ancestry Category')

    with pie1:
        st.markdown('### **Reference Panel Ancestry**')
        plot_pie(df_ancestry_counts)
        # st.dataframe(ref_combo)

    with pie3:
        st.markdown(f'### UKBB Predicted Ancestry')
        plot_pie(df_new_counts)
        # st.dataframe(new_combo)

    # Displays dataframe of Reference Panel Counts vs. Predicted Counts per Ancestry category
    with p2:
        st.dataframe(pie_table[['Ancestry Category', 'Ref Panel Counts', 'Predicted Counts']], hide_index=True, use_container_width=True)

with tabMethods:
    st.markdown("## _Ancestry_")
    st.markdown('### _Reference Panel_')
    st.markdown('The reference panel is composed of 4008 samples from 1000 Genomes Project, Human Genome Diversity Project (HGDP), and an Ashkenazi Jewish reference panel\
                (Gene Expression Omnibus (GEO) database, www.ncbi.nlm.nih.gov/geo (accession no. GSE23636)) (REF) with the following\
                ancestral makeup:')
    st.markdown(
                """
                - African (AFR): 819
                - African Admixed and Caribbean (AAC): 74
                - Ashkenazi Jewish (AJ): 471
                - Central Asian (CAS): 183
                - East Asian (EAS): 585
                - European (EUR): 534
                - Finnish (FIN): 99
                - Latino/American Admixed (AMR): 490
                - Middle Eastern (MDE): 152
                - South Asian (SAS): 601
                """
                )
    st.markdown('Samples were chosen from 1000 Genomes and HGDP to match the specific ancestries present in GP2. The reference panel was then\
                pruned for palindrome SNPs (A1A2= AT or TA or GC or CG). SNPs were then pruned for maf 0.05, geno 0.01, and hwe 0.0001.')

    st.markdown('### _Preprocessing_')
    st.markdown('The genotypes were pruned for geno 0.1. Common variants between the reference panel and the genotypes were extracted \
                from both the reference panel and the genotypes. Any missing genotypes were imputed using the mean of that particular\
                variant in the reference panel.')
    st.markdown('The reference panel samples were split into an 80/20 train/test set and then PCs were fit to and transformed the training\
                set using sklearn PCA.fit_transform and mean and standard deviation were extracted for scaling (𝑥/2) * (1 − (𝑥/2)) where 𝑥 is \
                the mean training set PCs. The test set was then transformed by the PCs in the training set using sklearn PCA.transform and\
                then normalized to the mean and standard deviation of the training set. Genotypes were then transformed by the same process as\
                the test set for prediction after model training.')

    st.markdown('### _UMAP + Classifier Training_')
    st.markdown('A classifier was then trained using UMAP transformations of the PCs and a extreme gradient boosting classifier (XGBoost) using a 5-fold\
                cross-validation using an sklearn pipeline and scored for balanced accuracy with a gridsearch over the following parameters:')
    st.markdown(
                """
                - “umap__n_neighbors”: [5,20]
                - “umap__n_components”: [15,25]
                - “umap__a”: [0.75, 1.0, 1.5]
                - “umap__b”: [0.25, 0.5, 0.75]
                - “xgb__C”: [0.001, 0.01, 0.1, 1, 10, 100]
                """
                )
    st.markdown('Performance varies from 95-98% balanced accuracy on the test set depending on overlapping genotypes.')

    st.markdown('### _Prediction_')
    st.markdown('Scaled PCs for genotypes are transformed using UMAP trained fitted by the training set and then predicted by the classifier. \
                    Genotypes are split and output into individual ancestries.')
