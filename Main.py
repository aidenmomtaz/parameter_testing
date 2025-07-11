# Run Command: streamlit run main.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from itertools import combinations
from tqdm import tqdm
import csv
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import os
import sys
import contextlib
from sklearn.metrics import confusion_matrix
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
import random

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def find_combinations(parameters):
    items = parameters
    all_combinations = []
    for r in range(1, len(items) + 1):
        all_combinations.extend(combinations(items, r))
    random.shuffle(all_combinations)
    return all_combinations

def predict(combinations, df, class_col, num_splits, classifier, percent_split, pos_class):
    # df[class_col] = df[class_col].fillna('healthy')
    df = df[df[class_col].notna()]
    y = df[class_col].to_numpy()

    classes = list(dict.fromkeys(y.tolist()))
    if pos_class not in classes:
        st.error("The selected positive class must be in the class column")
        st.stop()
    classes.remove(pos_class)
    classes.insert(0, pos_class)
    num_classes = len(classes)
    if num_classes < 2:
        st.error("You need two or more distinct classes to perform classification.")
        st.stop()
    clf_scores = pd.DataFrame(columns=['Acc','Precision','Sensitivity/TPR','Specificity/TNR','FPR','FNR'])

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)

    start_time = time.time()
    time_placeholder = st.empty()

    for i,combo in tqdm(enumerate(combinations)):
        acc_scores, prec_scores, sens_scores, spec_scores, fpr_scores, fnr_scores = [], [], [], [], [], []      
        for split in range(num_splits):
            X = df[list(combo)].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_split, stratify=y)

            # Set up and train model
            if classifier == 'Logistic':
                clf = LogisticRegression()
            elif classifier == 'Random Forest':
                clf = RandomForestClassifier()
            elif classifier == 'Multilayer Perceptron':
                clf = MLPClassifier(
                    hidden_layer_sizes=(32, 16, 8),
                    activation='relu',
                    solver='adam',
                    learning_rate='adaptive',
                    max_iter=500
                )
            elif classifier == 'XGBoost':
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(y_train)
                clf = XGBClassifier(
                    eval_metric='logloss'
                )
                clf.fit(X_train,y_encoded)
            elif classifier == 'LightGBM':
                feature_names = [f"f{i}" for i in range(X.shape[1])]
                X_train_df = pd.DataFrame(X_train, columns=feature_names)
                X_test_df = pd.DataFrame(X_test, columns=feature_names)

                with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
                    model = lgb.LGBMClassifier(verbose=-1)
                    model.fit(X_train_df, y_train)
                y_pred = model.predict(X_test_df)
            if classifier != 'XGBoost' and classifier != 'LightGBM':   
                clf.fit(X_train, y_train)

            # Predict
            if classifier != 'LightGBM':
                y_pred = clf.predict(X_test)
            if classifier == 'XGBoost':
                y_pred = encoder.inverse_transform(y_pred)

            # Calculate accuracy stats
            acc_scores.append(accuracy_score(y_test, y_pred))

            # Remap all non-pos_class labels to 'other' for multiclass classification
            if num_classes > 2:
                y_test_binary = [pos_class if y == pos_class else 'other' for y in y_test]
                y_pred_binary = [pos_class if y == pos_class else 'other' for y in y_pred]
                classes = [pos_class, 'other']
            else:
                y_test_binary = y_test
                y_pred_binary = y_pred

            # Calculate TPR, TNR, FPR, FNR
            prec_scores.append(precision_score(y_test_binary, y_pred_binary, pos_label=pos_class))
            cm = confusion_matrix(y_test_binary, y_pred_binary, labels=classes)
            tp, fn, fp, tn = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) else np.nan
            fpr = fp / (fp + tn) if (fp + tn) else np.nan
            fnr = fn / (fn + tp) if (fn + tp) else np.nan
            spec = tn / (tn + fp) if (tn + fp) else np.nan
            sens_scores.append(tpr)
            spec_scores.append(spec)
            fpr_scores.append(fpr)
            fnr_scores.append(fnr)


        # Add accuracy stats to an output dataframe
        clf_scores.loc[str(combo)] = [
            np.nanmean(acc_scores),
            np.nanmean(prec_scores),
            np.nanmean(sens_scores),
            np.nanmean(spec_scores),
            np.nanmean(fpr_scores),
            np.nanmean(fnr_scores)
        ]
        progress_bar.progress(int((i + 1) / len(combinations) * 100)) # increment progress bar

        # Calculate and output estimated remaining time
        if i % 50 == 0:
            elapsed = time.time() - start_time
            avg_per_iter = elapsed / (i + 1)
            est_total = avg_per_iter * len(combinations)
            est_remaining = est_total - elapsed
            if est_remaining < 60:
                time_placeholder.write(f"Iteration {i+1}/{len(combinations)} — Est. remaining: {est_remaining:.1f} seconds")
            else:
                est_remaining_min = est_remaining / 60
                time_placeholder.write(f"Iteration {i+1}/{len(combinations)} — Est. remaining: {est_remaining_min:.2f} minutes")
    
    # Clear progress bar and estimated remaining time
    progress_placeholder.empty()
    time_placeholder.empty()
    print(list(dict.fromkeys(y.tolist())))
    print(confusion_matrix(y_test, y_pred, labels=list(dict.fromkeys(y.tolist()))))

    # Output classifier label, stats dataframe, and total time taken
    st.subheader(f'{classifier} Classifier')
    st.dataframe(clf_scores)
    end_time = time.time()
    if (end_time - start_time) < 60:
        st.write(f"Total time: {(end_time - start_time):.2f} seconds")
    else:
        total_time_min = (end_time - start_time) / 60
        st.write(f"Total time: {total_time_min:.2f} minutes")


# UI code

st.title("Parameter Tester")

# Upload the file
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
if uploaded_file: # Output a preview of the data
    df = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded data:")      
    edited_df = st.data_editor(df)

# Get names of start, end, and class columns
start_col = st.text_input("Column name of your first parameter")
end_col = st.text_input("Column name of your last parameter")
class_col = st.text_input("Column name of the conditions")
if start_col and end_col and uploaded_file:
    # Find the column range
    start_idx = edited_df.columns.get_loc(start_col)
    end_idx = edited_df.columns.get_loc(end_col)
    col_range = edited_df.columns[start_idx:end_idx + 1] 
    col_range = [col for col in col_range if not edited_df[col].isnull().any()] # Filter out columns with None 

# Get name of positive class
pos_class = st.text_input("Name of the positive class")

# Get names of models 
models = ["Logistic","Random Forest","Multilayer Perceptron","XGBoost","LightGBM"]
selected_models = st.multiselect("Classifier models",models)

# Get the number of splits and percent to be test data
num_splits = st.number_input("Number of different train/test splits",min_value=1, max_value=200, step=1, value=20)
percent_split = st.number_input("Fraction of data you want to use for testing",min_value=0.05, max_value=0.5, step=0.01, value=0.15)

# Run the models when button is pressed and all fields are filled
run_button = st.button("Run")
if run_button:
    if uploaded_file and start_col and end_col and selected_models and num_splits and class_col and percent_split and pos_class:
        all_combinations = find_combinations(list(col_range)) # Get all combos of parameters
        for classifier in selected_models: # Run each model
            log_df = predict(all_combinations, edited_df, class_col, num_splits, classifier, percent_split, pos_class)
    else:
        st.warning('Please fill in all fields')
