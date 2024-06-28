import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class ThrombolysisChoiceOutcome():

    def __init__(self, rerun_models=True):
        """Constructor."""

        # Define fileds to use in models
        self.thrombolysis_choice_X_fields = [
            'stroke_team',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'arrival_to_scan_time',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'afib_anticoagulant',
            'age']

        self.thrombolysis_choice_y_field = 'thrombolysis'

        self.outcome_X_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis']

        self.outcome_y_field = 'discharge_disability'

        self.number_of_benchmark_hospitals = 25

        self.rerun_models = rerun_models

        # Utilities from Wang X, Moullaali TJ, Li Q, Berge E, Robinson TG, Lindley R, et al.
        # Utility-Weighted Modified Rankin Scale Scores for the Assessment of Stroke Outcome.
        # Stroke. 2020 Aug 1;51(8):2411-7

        self.mrs_utilities = np.array(
            [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

    def run(self):
        """Run the model."""
        self.load_data()
        if self.rerun_models:
            self.run_choice_model()
            self.run_outcome_model()
            self.patient_results.to_csv(
                './output/thrombolysis_choice_results.csv')

        self.analyse_results()

    def analyse_results(self):
        """Analyse patient level results"""

        # Load patient results
        self.patient_results = pd.read_csv(
            './output/thrombolysis_choice_results.csv', low_memory=False)

        # Average by stroke team
        results_by_team = self.patient_results.groupby('stroke_team').mean()
        results_by_team.drop('Unnamed: 0', axis=1, inplace=True)

        results_by_team['sensitivity'] = \
            results_by_team['TP'] / \
            (results_by_team['TP'] + results_by_team['FN'])
        results_by_team['specificity'] = \
            results_by_team['TN'] / \
            (results_by_team['TN'] + results_by_team['FP'])

        # Add stroke team ranks for sensitivity and specificity
        results_by_team['sensitivity_rank'] = results_by_team['sensitivity'].rank(
            ascending=False)
        results_by_team['specificity_rank'] = results_by_team['specificity'].rank(
            ascending=False)

        # Store results by team
        results_by_team.to_csv(
            './output/thrombolysis_outcome_predictions_by_team.csv')
        self.stroke_team_results = results_by_team

        # Create separate table of observed and benchmark thrombolysis rates
        thrombolysis_rates = self.stroke_team_results[[
            'thrombolysis', 'benchmark_decision', 'improved_outcome']]
        thrombolysis_rates.to_csv('./output/thrombolysis_rates.csv')

    def load_data(self):
        """Load required data for modelling."""

        # Load patient data (4 hour arrivals)
        self.data = pd.read_csv(
            './data/artificial_ml_data.csv', low_memory=False)

        # Add 20 mins simulated scan to thrombolysis time for all patients
        # To predict benefit opr not of thrombolysis if given 20 mins from scan
        self.data['simulated_onset_to_thrombolysis'] = (
            self.data['onset_to_arrival_time'] + self.data['arrival_to_scan_time'] + 20)
        
        # Artificial data is based on patients who have not received thrombectomy
        # For real data when training outcome model, remove patients who have received thrombectomy
        self.data['thrombectomy'] = 0

        # Make copy of data for results
        self.patient_results = self.data.copy(deep=True)

        # Get list of stroke teams
        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()

        
    def run_choice_model(self):
        """Train the model to predict thrombolysis choice."""

        # Get X and y
        X = self.data[self.thrombolysis_choice_X_fields]
        y = self.data[self.thrombolysis_choice_y_field]

        # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)

        # Define and Fit model
        self.choice_model = XGBClassifier(
            verbosity=0, seed=42, learning_rate=0.5)
        self.choice_model.fit(X_one_hot, y)

        # Get predictions
        y_pred_proba = self.choice_model.predict_proba(X_one_hot)[:, 1]
        y_pred = y_pred_proba >= 0.5
        self.patient_results['thrombolysis_choice_probability'] = y_pred_proba
        self.patient_results['thrombolysis_choice'] = y_pred * 1

        # Get accuracy
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f'ROC AUC: {roc_auc:0.3f}')
        print(f'Actual thrombolysis: {np.mean(y):0.3f}')
        print(f'Predicted thrombolysis: {np.mean(y_pred):0.3f}')

        # ************ HOSPITAL SHAP ************

        # Get hospital SHAP values for each patient
        explainer = shap.TreeExplainer(self.choice_model)
        shap_values_extended = explainer(X_one_hot)
        shap_values = shap_values_extended.values
        shap_values_df = pd.DataFrame(shap_values, columns=list(X_one_hot))

        # Sum hospital SHAPs for each patient
        shap_values_df['hospital'] = shap_values_df[self.stroke_teams].sum(
            axis=1)
        for team in self.stroke_teams:
            shap_values_df.drop(team, axis=1, inplace=True)

        # Add total SHAP to SHAP results for each patient
        shap_values_df['total'] = shap_values_df.sum(axis=1)

        # Get average hospital SHAP values
        shap_values_df['stroke_team'] = X['stroke_team'].values
        hospital_mean_shap = pd.DataFrame()
        hospital_mean_shap['hospital_SHAP'] = \
            shap_values_df.groupby('stroke_team').mean()['hospital']

        # Identify and label top 25 benchmark hospitals
        hospital_mean_shap.sort_values(
            by='hospital_SHAP', ascending=False, inplace=True)
        benchmark = np.zeros(len(hospital_mean_shap))
        benchmark[0:self.number_of_benchmark_hospitals] = 1
        hospital_mean_shap['benchmark'] = benchmark
        hospital_mean_shap.to_csv(
            './output/thrombolysis_choice_hospital_shap.csv')

        # Merge hospital_mean_shap values to patient results based on stroke team
        self.patient_results = pd.merge(
            self.patient_results, hospital_mean_shap, 
            left_on='stroke_team', right_index=True, how='left')
        self.patient_results.drop('benchmark', axis=1, inplace=True)

        # ************ Get benchmark decisions ************
        mask = hospital_mean_shap['benchmark'] == 1
        benchmark_hospitals = list(hospital_mean_shap[mask].index)
        decisions = []
        for benchmark_hosp in benchmark_hospitals:
            X_copy = X.copy(deep=True)
            X_copy['stroke_team'] = benchmark_hosp
            one_hot_encoded = encoder.transform(X_copy[['stroke_team']])
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded, columns=self.stroke_teams, index=X_copy.index)
            X_one_hot_copy = pd.concat([X_copy, one_hot_encoded_df], axis=1)
            X_one_hot_copy.drop('stroke_team', axis=1, inplace=True)
            decisions.append(self.choice_model.predict(X_one_hot_copy))
        # Get majority vote
        decisions = np.array(decisions)
        benchmark = decisions.mean(axis=0) >= 0.5
        benchmark = benchmark * 1
        self.patient_results['benchmark_decision'] = benchmark
        # Save
        shap_values_df.drop('stroke_team', axis=1, inplace=True)
        shap_values_df = shap_values_df.round(3)
        shap_values_df.to_csv('./output/thrombolysis_choice_shap.csv')

    def run_outcome_model(self):
        """
        """

        # For training remove patients who have received thrombectomy or who are haemorrhagic
        mask = (self.data['thrombectomy'] == 0) & (
            self.data['infarction'] == 1)
        train_data = self.data[mask]
        X_train = train_data[self.outcome_X_fields]
        y_train = train_data[self.outcome_y_field].values
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X_train[['stroke_team']])
        one_hot_encoded = encoder.transform(X_train[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=self.stroke_teams, index=X_train.index)
        X_train_one_hot = pd.concat([X_train, one_hot_encoded_df], axis=1)
        X_train_one_hot.drop('stroke_team', axis=1, inplace=True)

        # Define and Fit model
        self.outcome_model = XGBClassifier(
            verbosity=0, seed=42, learning_rate=0.5)
        self.outcome_model.fit(X_train_one_hot, y_train)

        # Get AUC
        y_probs = self.outcome_model.predict_proba(X_train_one_hot)
        auc = roc_auc_score(y_train.astype(np.int8), y_probs,
                            multi_class='ovo', average='macro')
        print(f'Outcome multiclass ROC AUC {auc:.3f}')

        # Predict all patients with and without thrombolysis
        X = self.data[self.outcome_X_fields]
        y = self.data[self.outcome_y_field].values
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)

        # Test with all onset_to_thrombolysis set to -10 (no thrombolysis)
        X_one_hot['onset_to_thrombolysis'] = -10
        all_patients_outcomes_untreated = self.outcome_model.predict_proba(
            X_one_hot)
        all_patients_outcomes_untreated_weighted_mrs = \
            (all_patients_outcomes_untreated * np.arange(7)).sum(axis=1)
        all_patients_outcomes_untreated_0_to_4 = all_patients_outcomes_untreated[:, 0:5].sum(
            axis=1)
        self.patient_results['untreated_weighted_mrs'] = 1.0 * \
            all_patients_outcomes_untreated_weighted_mrs
        self.patient_results['untreated_0_to_4'] = 1.0 * \
            all_patients_outcomes_untreated_0_to_4
        for i in range(7):
            self.patient_results[f'untreated_mrs_{i}'] = all_patients_outcomes_untreated[:, i]
        self.patient_results['untreated_utility'] = (
            all_patients_outcomes_untreated * self.mrs_utilities).sum(axis=1)

        # Test with all onset_to_thrombolysis set to simulated onset_to_thrombolysis
        X_one_hot['onset_to_thrombolysis'] = self.data['simulated_onset_to_thrombolysis']
        all_patients_outcomes_treated = self.outcome_model.predict_proba(
            X_one_hot)
        all_patients_outcomes_treated_weighted_mrs = \
            (all_patients_outcomes_treated * np.arange(7)).sum(axis=1)
        all_patients_outcomes_treated_0_to_4 = all_patients_outcomes_treated[:, 0:5].sum(
            axis=1)
        self.patient_results['treated_weighted_mrs'] = all_patients_outcomes_treated_weighted_mrs
        self.patient_results['treated_0_to_4'] = all_patients_outcomes_treated_0_to_4
        for i in range(7):
            self.patient_results[f'treated_mrs_{i}'] = all_patients_outcomes_treated[:, i]
        self.patient_results['treated_utility'] = (
            all_patients_outcomes_treated * self.mrs_utilities).sum(axis=1)

        # Compare treated and untreated outcomes
        self.patient_results['change_in_weighted_mrs'] = (
            all_patients_outcomes_treated_weighted_mrs - all_patients_outcomes_untreated_weighted_mrs)
        self.patient_results['change_in_mrs_0_to_4'] = all_patients_outcomes_treated_0_to_4 - \
            all_patients_outcomes_untreated_0_to_4

        # 'Improved outcome' is net improvement in mRS without an increase in mRS 5&6
        self.patient_results['improved_outcome'] = 1.0 * (
            (all_patients_outcomes_treated_weighted_mrs < all_patients_outcomes_untreated_weighted_mrs) &
            (all_patients_outcomes_treated_0_to_4 > all_patients_outcomes_untreated_0_to_4))

        # Calculate change in utility
        self.patient_results['change_in_utility'] = self.patient_results['treated_utility'] - \
            self.patient_results['untreated_utility']

        # Compare outcome with thrombolysis given
        self.patient_results['thrombolysis_given_agrees_with_improved_outcome'] = 1.0 * (
            self.patient_results['thrombolysis'] == self.patient_results['improved_outcome'])
        self.patient_results['TP'] = 1.0 * (
            (self.patient_results['thrombolysis'] == 1) &
            (self.patient_results['improved_outcome'] == 1))
        self.patient_results['FP'] = 1.0 * (
            (self.patient_results['thrombolysis'] == 1) &
            (self.patient_results['improved_outcome'] == 0))
        self.patient_results['FN'] = 1.0 * (
            (self.patient_results['thrombolysis'] == 0) & 
            (self.patient_results['improved_outcome'] == 1))
        self.patient_results['TN'] = 1.0 * (
            (self.patient_results['thrombolysis'] == 0) &
            (self.patient_results['improved_outcome'] == 0))

        # Delete outcome results for when infarction = 0
        mask = self.data['infarction'] == 0
        self.patient_results.loc[mask, 'untreated_weighted_mrs'] = np.nan
        self.patient_results.loc[mask, 'untreated_0_to_4'] = np.nan
        self.patient_results.loc[mask, 'treated_weighted_mrs'] = np.nan
        self.patient_results.loc[mask, 'treated_0_to_4'] = np.nan
        self.patient_results.loc[mask, 'change_in_weighted_mrs'] = np.nan
        self.patient_results.loc[mask, 'change_in_mrs_0_to_4'] = np.nan
        self.patient_results.loc[mask, 'improved_outcome'] = 0
        self.patient_results.loc[mask, 'untreated_utility'] = np.nan
        self.patient_results.loc[mask, 'treated_utility'] = np.nan
        self.patient_results.loc[mask, 'change_in_utility'] = np.nan
        for i in range(7):
            self.patient_results.loc[mask, f'untreated_mrs_{i}'] = np.nan
            self.patient_results.loc[mask, f'treated_mrs_{i}'] = np.nan
        self.patient_results.loc[mask,
                                 'thrombolysis_given_agrees_with_improved_outcome'] = np.nan
        self.patient_results.loc[mask, 'TP'] = np.nan
        self.patient_results.loc[mask, 'FP'] = np.nan
        self.patient_results.loc[mask, 'FN'] = np.nan
        self.patient_results.loc[mask, 'TN'] = np.nan
