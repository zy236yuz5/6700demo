import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import joblib
import plotly.express as px

#Please note that this is the streamlit APP ！！
# Please run from the root directory!
# use "streamlit run project-3/app.py" to run it 

# set it for wide layout
st.set_page_config(layout="wide")

# Below is the preset for the class object used in our stacked model
class StackedClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models  # It will be a list since we use multiple base models here
        self.meta_model = meta_model
        self.predictions = None  # Initialize as None; we'll set it in fit_with_cv
        self.accuracies = []  # Store accuracies

    # do  cross validation
    def fit_with_cv(self, X, y, cv_folds=5):
        n_samples, n_models = len(X), len(self.base_models)

        # start the  matrix to store predictions from each base model for each sample
        self.predictions = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=1789)
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        for train_index, val_index in kf.split(X):
            fold_X_train, fold_X_val = X[train_index], X[val_index]
            fold_y_train, fold_y_val = y[train_index], y[val_index]

            fold_predictions = np.zeros((len(val_index), n_models))

            for idx, model in enumerate(self.base_models):
                # fit base model on the training portion of the fold
                model.fit(fold_X_train, fold_y_train)

                # predict using val set
                fold_y_pred = model.predict(fold_X_val)
                fold_predictions[:, idx] = fold_y_pred
                self.predictions[val_index, idx] = fold_y_pred

            # fit meta model on this fold's predictions and compute accuracy
            self.meta_model.fit(fold_predictions, fold_y_val)
            aggregated_predictions = self.meta_model.predict(fold_predictions)
            fold_accuracy = accuracy_score(fold_y_val, aggregated_predictions)
            self.accuracies.append(fold_accuracy)

        # fit our meta model based on all previous predictions
        self.meta_model.fit(self.predictions, y)


    #  prediction function
    def predict(self, X):
        meta_input = np.column_stack([
            model.predict(X) for model in self.base_models
        ])

        # now take the meta model and get the final predictions
        final_predictions = self.meta_model.predict(meta_input)

        return final_predictions

    def get_feature_importance(self, X):
        # double check if meta-model supports feature_importances_
        if not hasattr(self.meta_model, 'feature_importances_'):
            raise ValueError("Meta-model doesn't support feature_importances_ attribute.")

        # remember importance of each base model's predictions in meta-model
        meta_importances = self.meta_model.feature_importances_

        # re start  an array to store the cumulative feature importances
        cumulative_importances = np.zeros(X.shape[1])

        # compute each base model feature importance
        for idx, model in enumerate(self.base_models):
            if hasattr(model, 'feature_importances_'):
                # Weight the feature importance of this base model by its importance in the meta-model
                weighted_importance = model.feature_importances_ * meta_importances[idx]
                cumulative_importances += weighted_importance

        # normalization for the importances so they sum up to 1
        cumulative_importances = cumulative_importances / np.sum(cumulative_importances)

        return cumulative_importances



# Load the trained model
model_filename = 'project-3/stacked_model_revised.pkl'
stacked_clf = joblib.load(model_filename)

# Load the base feature importance plot HTML content
with open('project-3/feature_importance_base.html', 'r') as f:
    base_feature_importance_plot = f.read()

# Streamlit UI
st.title('Bank Term Deposit Subscription Prediction')

# User input
features_input = st.text_input('Enter features separated by commas (Press Enter to Apply): ')
if features_input:
    try:
        features = np.fromstring(features_input, sep=',')
        features = features.reshape(1, -1)
        prediction = stacked_clf.predict(features).tolist()[0]
        
        # Translate the prediction to a comprehensible message
        if prediction == 1:
            result_message = "The customer is predicted to subscribe to a term deposit."
        else:
            result_message = "The customer is not predicted to subscribe to a term deposit."
        
        st.success(result_message)
        
    except Exception as e:
        st.error(f'Error: {str(e)}')


# Display base feature importance plot
st.markdown('## Base Feature Importance')

centered_html = f"""
<div style="display: flex; justify-content: center; align-items: center; ">
    {base_feature_importance_plot}
</div>
"""
st.components.v1.html(centered_html, height=800,width=1200)
