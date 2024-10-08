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

    def __init__(self, data_path):
        """Constructor."""

        self.data_path = data_path

        # Define fields to use in models
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

        # Utilities from Wang X, Moullaali TJ, Li Q, Berge E, Robinson TG, Lindley R, et al.
        # Utility-Weighted Modified Rankin Scale Scores for the Assessment of Stroke Outcome.
        # Stroke. 2020 Aug 1;51(8):2411-7

        self.mrs_utilities = np.array(
            [0.97, 0.88, 0.74, 0.55, 0.20, -0.19, 0.00])

    def run(self):
        """Run the model."""
        self.load_data()
        self.run_choice_model()
        self.run_outcome_model()
        self.patient_results.to_csv('./output/thrombolysis_choice_results.csv')
        self.analyse_results()
        self.predict_prototype_patients_all_teams()

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
            f'{self.data_path}/ml_data.csv', low_memory=False)

        # Add 20 mins simulated scan to thrombolysis time for all patients
        # To predict benefit opr not of thrombolysis if given 20 mins from scan
        self.data['simulated_onset_to_thrombolysis'] = (
            self.data['onset_to_arrival_time'] + self.data['arrival_to_scan_time'] + 20)
        
        # Make copy of data for results
        self.patient_results = self.data.copy(deep=True)

        # Get list of stroke teams
        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()

        # Load prototype patient data
        self.prototype_patients = pd.read_csv(
            f'{self.data_path}/prototype_patients.csv', index_col='Patient prototype')
        

    def predict_prototype_patients_all_teams(self):
        """
        Predict thrombolysis choice for prototype patients for all stroke teams.

        Uses thrombolysis choice model fitted in `run_choice_model`

        * Get thrombolysis use benchmark prediction for each prototype patient
            (majority vote of benchmark hospitals)
        * Get thrombolysis use prediction for each prototype patient at each stroke team
        * Report thtombolysis use as a percentage (based on probability of receiving thrombolysis; 
            this will also be the proportion of those patients who are likely to receive thrombolysis.
        * Save as `prototype_patients_all_teams.csv`
        """

        results = pd.DataFrame(index=self.prototype_patients.index)

        # Set up encoder
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)

        # Get benchmark predictions
        decisions = []
        for benchmark_hosp in self.benchmark_hospitals:
            prototype_patients = self.prototype_patients.copy(deep=True)
            prototype_patients['stroke_team'] = benchmark_hosp
            # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
            encoder.fit(prototype_patients[['stroke_team']])
            one_hot_encoded = encoder.transform(prototype_patients[['stroke_team']])
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded, columns=self.stroke_teams, index=prototype_patients.index)
            X_one_hot = pd.concat([prototype_patients, one_hot_encoded_df], axis=1)
            X_one_hot.drop('stroke_team', axis=1, inplace=True)
            decisions.append(self.choice_model.predict_proba(X_one_hot)[:, 1])
        decisions = np.array(decisions)
        results['benchmark'] = decisions.mean(axis=0)

        # Get predictions for all stroke teams
        for stroke_team in self.stroke_teams:
            prototype_patients = self.prototype_patients.copy(deep=True)
            prototype_patients['stroke_team'] = stroke_team
            # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
            encoder.fit(prototype_patients[['stroke_team']])
            one_hot_encoded = encoder.transform(prototype_patients[['stroke_team']])
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded, columns=self.stroke_teams, index=prototype_patients.index)
            X_one_hot = pd.concat([prototype_patients, one_hot_encoded_df], axis=1)
            X_one_hot.drop('stroke_team', axis=1, inplace=True)
            # Get predictions from self.choice_model
            y_pred_proba = self.choice_model.predict_proba(X_one_hot)[:, 1]
            results[stroke_team] = y_pred_proba

        # Save (make percentage, transpose, rename empty column name)
        results = results * 100
        results = results.T
        results.index.name = 'stroke_team'
        results.to_csv('./output/prototype_patients_all_teams.csv')


    def predict_prototype_patients_single_team(self, stroke_team, anon=True, show=False, save=False):

        team_name = "Anonymous" if anon else stroke_team
        prototype_patients = self.prototype_patients.copy(deep=True)
        prototype_patients['stroke_team'] = stroke_team
        # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(prototype_patients[['stroke_team']])
        one_hot_encoded = encoder.transform(prototype_patients[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=self.stroke_teams, index=prototype_patients.index)
        X_one_hot = pd.concat([prototype_patients, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)
        # Get predictions from self.choice_model
        y_pred_proba = self.choice_model.predict_proba(X_one_hot)[:, 1]
        # Get benchmark predictions
        decisions = []
        for benchmark_hosp in self.benchmark_hospitals:
            X_copy = prototype_patients.copy(deep=True)
            X_copy['stroke_team'] = benchmark_hosp
            one_hot_encoded = encoder.transform(X_copy[['stroke_team']])
            one_hot_encoded_df = pd.DataFrame(
                one_hot_encoded, columns=self.stroke_teams, index=X_copy.index)
            X_one_hot_copy = pd.concat([X_copy, one_hot_encoded_df], axis=1)
            X_one_hot_copy.drop('stroke_team', axis=1, inplace=True)
            decisions.append(self.choice_model.predict_proba(X_one_hot_copy)[:, 1])
        # Get majority vote
        decisions = np.array(decisions)
        benchmark = decisions.mean(axis=0)

        # Put results in DataFrame
        benchmark_results = pd.DataFrame()
        benchmark_results['Benchmark'] = benchmark * 100
        benchmark_results[f'{team_name}'] = y_pred_proba * 100

        # Plot as vertical bar chart
        labels = [i.replace('+', '+\n') for i in list(self.prototype_patients.index)]
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(111)
        benchmark_results.plot.bar(ax=ax)
        ax.set_ylim(0,100)
        ax.set_ylabel('% Patients likely to receive thrombolysis')
        # rebuild the xticklabels
        ax.set_xticklabels(labels, rotation=90)
        ax.grid(axis = 'y')


        # Save & close
        if save:
            plt.savefig(f'./output/prototype_patients_{stroke_team}.png', bbox_inches='tight')


        if show:
            plt.show()
        
        plt.close()

    def run_choice_model(self):
        """
        Train a model to predict thrombolysis choice.
   
        * Get X and y data
        * One-hot encode stroke teams
        * Fit XGBoost model on all data (`learning_rate=0.5` prevents loss of effect of spare stroke team features)
        * Get predictions of y, and assess accuracy
        * Get hospital SHAP for each patients, and average by stroke team
        * Identify benchmark hospitals (by hospital SHAP)
        * Save hopsital SHAP values to `thrombolysis_choice_hospital_shap.csv`
        * Get benchmark decisions for each patient
        * Save all patients SHAP values to `thrombolysis_choice_shap.csv`        
        """

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
        self.choice_model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        self.choice_model.fit(X_one_hot, y)

        # Get predictions
        y_pred_proba = self.choice_model.predict_proba(X_one_hot)[:, 1]
        y_pred = y_pred_proba >= 0.5
        self.patient_results['thrombolysis_choice_probability'] = y_pred_proba
        self.patient_results['thrombolysis_choice'] = y_pred * 1

        # Get accuracy
        fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print('\nAccuracy scores are for guidance only; all data is used to fit model')
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
        shap_values_df['hospital'] = shap_values_df[self.stroke_teams].sum(axis=1)
        for team in self.stroke_teams:
            shap_values_df.drop(team, axis=1, inplace=True)

        # Add total SHAP to SHAP results for each patient
        shap_values_df['total'] = shap_values_df.sum(axis=1)

        # Get average hospital SHAP values
        shap_values_df['stroke_team'] = X['stroke_team'].values
        hospital_mean_shap = pd.DataFrame()
        hospital_mean_shap['hospital_SHAP'] = \
            shap_values_df.groupby('stroke_team').mean()['hospital']

        # Identify and label top benchmark hospitals
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
        self.benchmark_hospitals = list(hospital_mean_shap[mask].index)
        decisions = []
        for benchmark_hosp in self.benchmark_hospitals:
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
        benchmark = decisions.mean(axis=0)
        benchmark = benchmark * 1
        self.patient_results['benchmark_decision'] = benchmark
        # Save
        shap_values_df.drop('stroke_team', axis=1, inplace=True)
        shap_values_df = shap_values_df.round(3)
        shap_values_df.to_csv('./output/thrombolysis_choice_shap.csv')

    def run_outcome_model(self):
        """
        Train a model to predict outcomes (probabilities of discharge mRS).
        The model is trained only on infractions stroke patients who did not also have thrombectomy.
        Predictions of outcomes are made for all patients, but are removed for non-infarction stroke patients.

        * Get X and y data
        * One-hot encode stroke teams
        * Fit XGBoost model on all data (`learning_rate=0.5` prevents loss of effect of spare stroke team features)
        * Get predictions of outcomes (mRS probabilities), and assess accuracy
        * Predict outcomes all patients with and without thrombolysis
            * Use onset to thrombolysis of 99999 when no thrombolysis
            * Use simulated_onset_to_thrombolysis for use of thrombolysis
        * Calculate further outcome results from mRS probabilities:
            * Probability weighted mRS (untreated, treated, difference) 
            * Proportion mRS 0-4  (untreated, treated, difference) 
            * Utility  (untreated, treated, difference) 
            * Improved outcome (improved probability weighted mRS **and** improved proportion mRS 0-4)
        * Compare outcomes with observed use of thrombolysis:
            * TP (true positive) = thrombolysis given and predicted improved outcome
            * FP (false positive) = thrombolysis given and predicted not improved outcome
            * FN (false negative) = thrombolysis not given and predicted improved outcome
            * TN (true negative) = thrombolysis not given and predicted not improved outcome
        * Delete outcomes for non-infraction stroke
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
        self.outcome_model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        self.outcome_model.fit(X_train_one_hot, y_train)

        # Get AUC
        y_probs = self.outcome_model.predict_proba(X_train_one_hot)
        auc = roc_auc_score(y_train.astype(np.int8), y_probs,
                            multi_class='ovo', average='macro')
        print('\nAccuracy score is for guidance only; all data is used to fit model')
        print(f'Outcome multiclass ROC AUC {auc:.3f}')

        # Predict outcomes all patients with and without thrombolysis
        X = self.data[self.outcome_X_fields]
        y = self.data[self.outcome_y_field].values
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(
            one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)

        # Test with all onset_to_thrombolysis set to 99999 (no thrombolysis)
        X_one_hot['onset_to_thrombolysis'] = 99999
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
