import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
import scipy.stats
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

class IndividualPatientModel:

    """
    Class for individual patient models (thrombolysis choice and outcome).
    """

    def __init__(self, data_path, train_models=False, replicates=30):
        """
        Initialize the class.
        """

        self.data_path = data_path

        self.thrombolysis_choice_fields = [
            'stroke_team',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'arrival_to_scan_time',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'afib_anticoagulant',
            'age',
            'thrombolysis'
        ]

        self.thrombolysis_outcome_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis',
            'discharge_disability'
        ]

        self.data = pd.read_csv(f'{data_path}/ml_data.csv', low_memory=False)    

        # Set up one hot encoder
        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()
        enc = OneHotEncoder(categories=[self.stroke_teams])
        
        # Get thrombolysis data
        thrombolysis_data = self.data[self.thrombolysis_choice_fields]
        one_hot = enc.fit_transform(thrombolysis_data[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        thrombolysis_data = pd.concat([thrombolysis_data, one_hot], axis=1)
        self.thrombolysis_data = thrombolysis_data.drop(columns=['stroke_team'])

        # Get stroke outcome data       
        outcome_data = self.data.copy()

        # Only train outcome model when no thrombectomy given and for infarction patients
        mask = (outcome_data['thrombectomy'] == 0) & (outcome_data['infarction'] == 1)
        outcome_data = outcome_data[mask]

        # Restrict fields
        outcome_data = outcome_data[self.thrombolysis_outcome_fields]

        # Remove empty rows
        self.outcome_data = outcome_data.dropna()
        
        outcome_data = pd.concat([outcome_data, one_hot], axis=1)
        self.outcome_data = outcome_data.drop(columns=['stroke_team'])

        # Get benchmark data
        benchmark_data = pd.read_csv(
            './output/thrombolysis_choice_hospital_shap.csv')        
        mask = benchmark_data['benchmark'] == 1
        benchmark_data = benchmark_data[mask]
        self.benchmark_hospitals = benchmark_data['stroke_team'].values

        # Train new models or load existing models
        if train_models:
            self.train_models(replicates)
        
        # Load models
        self.choice_models = pickle.load(
            open('./pickled_models/replicate_choice_models.pkl', 'rb'))
        self.outcome_models = pickle.load(
            open('./pickled_models/replicate_outcome_models.pkl', 'rb'))


    def plot_patient_results(self, patient, save, filename, anon):

        fig = plt.figure(figsize=(15, 6))

        # Add patient dictionary as a text box
        ax = fig.add_subplot(131)
        patient_dict = patient.iloc[0].to_dict()
        if anon:
            patient_dict['stroke_team'] = 'ANONYMOUS'

        patient_text = 'PATIENT CHARACTERISTICS\n\n'

        patient_text = patient_text + (
            '\n'.join([f'{k}: {v}' for k, v in patient_dict.items()]))
        

        # Add thrombolysis choice prediction
        patient_text += f'\n\n\nTHROMBOLYSIS (IVT) CHOICE:'
        patient_text += 'Of 100 patients like\nthis, how many would receive IVT:\n\n'
        prediction = np.round(self.thrombolysis_prediction * 100, 0)
        patient_text += f'This hospital = {prediction:0.0f}\n'
        prediction = np.round(self.thrombolysis_choice_benchmark_mean * 100, 0)
        patient_text += f'Benchmark hospitals = {prediction:0.0f}\n'

        patient_text = patient_text + f'\n\nLIKELY OUTCOME (mean ± 95% confidence interval)\n'

        v = self.untreated_weighted_mrs
        ci = self.untreated_weighted_mrs_ci
        patient_text += f'\nUntreated weighted mRS = {v:0.2f} ({ci:0.2f})'
        v = self.treated_weighted_mrs
        ci = self.treated_weighted_mrs_ci
        patient_text += f'\nTreated weighted mRS = {v:0.2f} ({ci:0.2f})'
        v = self.improvement
        c1 = self.improvement_ci
        patient_text += f'\nmRS improvement due to IVT = {v:0.2f} ({c1:0.2f})'

        v = self.untreated_less_3
        c1 = self.untreated_less_3_ci
        patient_text += f'\n\nUntreated proportion mRS 0-2 = {v:0.2f} ({c1:0.2f})'
        v = self.treated_less_3
        c1 = self.treated_less_3_ci
        patient_text += f'\nTreated proportion mRS 0-2 = {v:0.2f} ({c1:0.2f})'
        v = self.change_in_less_3
        c1 = self.change_in_less_3_ci
        patient_text += f'\nChange in proportion mRS 0-2 due to IVT = {v:0.2f} ({c1:0.2f})'
        if v > 0:
            nnt = int(np.round(1/v, 0))
            patient_text += '\nBenefit in mRS 0-2 due to IVT'
            patient_text += f'\nNumber needed to treat (for additional mRS 0-2) = {nnt}'
        else:
            nnt = 0 - int(np.round(1/v, 0))
            patient_text += '\nHarm in mRS 0-2 due to IVT'
            patient_text += f'\nNumber needed to treat (for reduced mRS 0-2) = {nnt}'

        v = self.untreated_more_4
        c1 = self.untreated_more_4_ci
        patient_text += f'\n\nUntreated proportion mRS 5-6 = {v:0.2f} ({c1:0.2f})'
        v = self.treated_more_4
        c1 = self.treated_more_4_ci
        patient_text += f'\nTreated proportion mRS 5-6 = {v:0.2f} ({c1:0.2f})'
        v = self.change_in_more_4
        c1 = self.change_in_more_4_ci
        patient_text += f'\nChange in proportion mRS 5-6 due to IVT = {v:0.2f} ({c1:0.2f})'
        if v < 0:
            nnt = 0 - int(np.round(1/v, 0))
            patient_text += '\nBenefit in mRS 5-6 due to IVT'
            patient_text += f'\nNumber needed to treat (for avoided mRS 5-6) = {nnt}'
        else:
            nnt = int(np.round(1/v, 0))
            patient_text += '\nHarm in mRS 5-6 due to IVT'
            patient_text += f'\nNumber needed to treat (for additional mRS 5-6) = {nnt}'

        ax.text(0.02, 1.07, patient_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top')

        # Remove all axes
        ax.axis('off')

        # Plot outcomes
        ax = fig.add_subplot(132)
        x = np.arange(7)
        ax.bar(x-0.2, self.untreated_dist, 
            color='red', label=f'Untreated', linewidth=1, linestyle='--', width=0.4, alpha=0.7)
        ax.bar(x+0.2, self.treated_dist,
            color='blue', label=f'Treated', linewidth=1, linestyle='--', width=0.4, alpha=0.7)
        ax.errorbar(x-0.2, self.untreated_dist, yerr=self.untreated_dist_ci, fmt='none',
                    ecolor='black', capsize=2)
        ax.errorbar(x+0.2, self.treated_dist, yerr=self.treated_dist_ci, fmt='none',
                    ecolor='black', capsize=2)

        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.legend()
        ax.set_xlabel('Discharge disability (mRS)')
        ax.set_ylabel('Probability')
        ax.set_title('Discharge disability\nprobability distribution')


        # Plot cumulative values for treated and untreated
        ax = fig.add_subplot(133)
        x = np.arange(7)
        untreated_cum = np.cumsum(self.untreated_dist)
        treated_cum = np.cumsum(self.treated_dist)
        ax.plot(x, untreated_cum, color='red', label=f'Untreated', linewidth=1, linestyle=':',
                alpha=0.7)
        ax.plot(x, treated_cum, color='blue', label=f'Treated', linewidth=1, linestyle='--',
                alpha=0.7)
        # Fil the difference between the lines
        ax.fill_between(x, untreated_cum, treated_cum, where=treated_cum >= untreated_cum, 
                        facecolor='blue', interpolate=True, alpha=0.2)
        ax.fill_between(x, untreated_cum, treated_cum, where=treated_cum <= untreated_cum,
                        facecolor='red', interpolate=True, alpha=0.2)
        ax.legend()
        ax.set_xlabel('Discharge disability (mRS)')
        ax.set_ylabel('Cumulative probability')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.set_title('Cumulative probability\nof discharge disability')

        txt  = 'Shaded area:\nBlue: Treated better\nRed: Untreated better'
        ax.text(0.48, 0.05, txt, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        # Add gaps between figures
        plt.subplots_adjust(wspace=0.3)
        plt.close()

        # Save to patient_output folder if required
        if save:
            fig.savefig(f'./patient_output/{filename}.png', dpi=300)

        # Store figure
        self.results_fig = fig


    def predict_patient(
            self, patient_data, save=False, filename=None, anon=False):

        patient = pd.DataFrame(patient_data, index=[0])

        # Get thrombolysis choice prediction
        fields = self.thrombolysis_choice_fields.copy()
        fields.remove('thrombolysis')
        patient_choice = patient[fields]
        enc = OneHotEncoder(categories=[self.stroke_teams])
        one_hot = enc.fit_transform(patient_choice[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        patient_choice = pd.concat([patient_choice, one_hot], axis=1)
        patient_choice.drop('stroke_team', axis=1, inplace=True)
        thrombolysis_predictions = []
        for i in range(len(self.choice_models)):
            model = self.choice_models[i]
            thrombolysis_predictions.append(model.predict_proba(patient_choice)[:,1])
        thrombolysis_predictions = np.array(thrombolysis_predictions)
        self.thrombolysis_prediction = np.mean(thrombolysis_predictions)
        self.thrombolysis_prediction_std = np.std(thrombolysis_predictions)
        sem = self.thrombolysis_prediction_std / np.sqrt(len(thrombolysis_predictions))
        self.thrombolysis_prediction_ci = \
            sem * scipy.stats.t.ppf((1 + 0.95) / 2., len(thrombolysis_predictions)-1)

        # Get benchmark thrombolysis predictions
        benchmark_predictions = []
        for benchmark_hosp in self.benchmark_hospitals:
            p = patient_choice.copy()
            # Change one-hot encoding
            current_team = patient_data['stroke_team']
            p[f'{current_team}'] = False
            p[f'{benchmark_hosp}'] = True
            # Get predictions
            thrombolysis_predictions = []
            for i in range(len(self.choice_models)):
                model = self.choice_models[i]
                thrombolysis_predictions.append(model.predict_proba(p)[:,1])
            # Reset hospital
            p[f'{benchmark_hosp}'] = False
            # Get mean prediction
            thrombolysis_predictions = np.array(thrombolysis_predictions)
            benchmark_prediction = np.mean(thrombolysis_predictions)
            benchmark_predictions.append(benchmark_prediction)
        
        self.thrombolysis_choice_benchmark_mean = np.mean(benchmark_predictions)
        self.thrombolysis_choice_benchmark_std = np.std(benchmark_predictions)
        sem = self.thrombolysis_choice_benchmark_std / np.sqrt(len(benchmark_predictions))
        self.thrombolysis_choice_benchmark_ci = \
            sem * scipy.stats.t.ppf((1 + 0.95) / 2., len(benchmark_predictions)-1)

        # Get thrombolysis outcome prediction
        untreated_dist = []
        treated_dist = []
        untreated_less_3 = []
        treated_less_3 = []
        untreated_more_4 = []
        treated_more_4 = []
        untreated_weighted_mrs = []
        treated_weighted_mrs = []
        improvement = []
        fields = self.thrombolysis_outcome_fields.copy()
        fields.remove('discharge_disability')
        p = patient[fields]
        enc = OneHotEncoder(categories=[self.stroke_teams])
        one_hot = enc.fit_transform(p[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        p_treated = pd.concat([p, one_hot], axis=1)
        p_treated.drop('stroke_team', axis=1, inplace=True)
        p_untreated = p_treated.copy()
        p_untreated['onset_to_thrombolysis'] = 99999
    
        for i in range(len(self.outcome_models)):        
        # Get untreated and treated distributions
            untreated = self.outcome_models[i].predict_proba(p_untreated).flatten()
            treated = self.outcome_models[i].predict_proba(p_treated).flatten()
            untreated_dist.append(untreated)
            treated_dist.append(treated)
            # Get weighted average of mRS scores
            weighted_untreated = np.sum(untreated * np.arange(7))
            weighted_treated = np.sum(treated * np.arange(7))
            untreated_weighted_mrs.append(weighted_untreated)
            treated_weighted_mrs.append(weighted_treated)
            improvement.append(0-(weighted_treated - weighted_untreated))
            # Get untreated and treated distributions for mRS <3
            untreated_less_3.append(np.sum(untreated[:3]))
            treated_less_3.append(np.sum(treated[:3]))
            # Get untreated and treated distributions for mRS >4
            untreated_more_4.append(np.sum(untreated[5:]))
            treated_more_4.append(np.sum(treated[5:]))

        # Get mean distribution predictions
        untreated_dist = np.array(untreated_dist)
        treated_dist = np.array(treated_dist)
        self.untreated_dist = np.mean(untreated_dist, axis=0)
        self.treated_dist = np.mean(treated_dist, axis=0)
        self.untreated_dist_std = np.std(untreated_dist, axis=0)
        self.treated_dist_std = np.std(treated_dist, axis=0)
        n = len(self.outcome_models)
        sem = self.untreated_dist_std / np.sqrt(n)
        self.untreated_dist_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        sem = self.treated_dist_std / np.sqrt(n)
        self.treated_dist_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        # Get mean predictions for mRS <3
        untreated_less_3 = np.array(untreated_less_3)
        treated_less_3 = np.array(treated_less_3)
        self.untreated_less_3 = np.mean(untreated_less_3)
        self.treated_less_3 = np.mean(treated_less_3)
        self.untreated_less_3_std = np.std(untreated_less_3)
        self.treated_less_3_std = np.std(treated_less_3)
        sem = self.untreated_less_3_std / np.sqrt(n)
        self.untreated_less_3_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        sem = self.treated_less_3_std / np.sqrt(n)
        self.treated_less_3_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        # Get change in proportion mRS <3
        self.change_in_less_3 = np.mean(self.treated_less_3 - self.untreated_less_3)
        self.change_in_less_3_std = np.std(self.treated_less_3 - self.untreated_less_3)
        sem = self.change_in_less_3_std / np.sqrt(n)
        self.change_in_less_3_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)         
        # Get mean predictions for mRS >4
        untreated_more_4 = np.array(untreated_more_4)
        treated_more_4 = np.array(treated_more_4)
        self.untreated_more_4 = np.mean(untreated_more_4)
        self.treated_more_4 = np.mean(treated_more_4)
        self.untreated_more_4_std = np.std(untreated_more_4)
        self.treated_more_4_std = np.std(treated_more_4)
        sem = self.untreated_more_4_std / np.sqrt(n)
        self.untreated_more_4_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)        
        sem = self.treated_more_4_std / np.sqrt(n)
        self.treated_more_4_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        # Get change in proportion mRS >4
        self.change_in_more_4 = np.mean(self.treated_more_4 - self.untreated_more_4)
        self.change_in_more_4_std = np.std(self.treated_more_4 - self.untreated_more_4)
        sem = self.change_in_more_4_std / np.sqrt(n)
        self.change_in_more_4_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        # Get mean predictions for weighted mRS
        untreated_weighted_mrs = np.array(untreated_weighted_mrs)
        treated_weighted_mrs = np.array(treated_weighted_mrs)
        self.untreated_weighted_mrs = np.mean(untreated_weighted_mrs)
        self.treated_weighted_mrs = np.mean(treated_weighted_mrs)
        self.untreated_weighted_mrs_std = np.std(untreated_weighted_mrs)
        self.treated_weighted_mrs_std = np.std(treated_weighted_mrs)
        sem = self.untreated_weighted_mrs_std / np.sqrt(n)
        self.untreated_weighted_mrs_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        sem = self.treated_weighted_mrs_std / np.sqrt(n)
        self.treated_weighted_mrs_ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        # Get mean predictions for improvement
        improvement = np.array(improvement)
        self.improvement = np.mean(improvement)
        self.improvement_std = np.std(improvement)
        sem = self.improvement_std / np.sqrt(n)
        self.improvement_ci= sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)        
        
        # Call plotting function
        self.plot_patient_results(patient, save, filename, anon)
        return self.results_fig
        

    def train_models(self, replicates):
        """
        Train and save the models.
        """
        
        # THROMBOLYSIS CHOICE MODELS
        print('Training thrombolysis choice models...')

        # Fit models
        model_full = []
        for i in range(replicates):
            # Sample data
            sample = self.thrombolysis_data.sample(frac=1.0, random_state=42+i, replace=True)
            X = sample.drop(columns=['thrombolysis'])
            y = sample['thrombolysis']
            # Fit full model
            model = XGBClassifier(random_state=42+i, learning_rate=0.5)
            model.fit(X, y)
            model_full.append(model)
        # Pickle models
        pickle.dump(model_full, open('./pickled_models/replicate_choice_models.pkl', 'wb'))

        # THROMBOLYSIS OUTCOME MODELS
        print('Training thrombolysis outcome models...')

        # Fit models
        model_full = []
        for i in range(replicates):
            # Sample data
            sample = self.outcome_data.sample(frac=1.0, random_state=42+i, replace=True)

            # remove any with y <0 or > 6
            sample = sample[(sample['discharge_disability'] >= 0) & (sample['discharge_disability'] <= 6)]

            X = sample.drop(columns=['discharge_disability'])
            y = sample['discharge_disability'].values
            y = y.astype(int)
            # Fit full model
            model = XGBClassifier(random_state=42+i)
            model.fit(X, y)
            model_full.append(model)
        # Pickle models
        pickle.dump(model_full, open('./pickled_models/replicate_outcome_models.pkl', 'wb'))
