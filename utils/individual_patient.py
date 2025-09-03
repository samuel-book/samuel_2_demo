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
        outcome_data = outcome_data.dropna()

        # One hot encode stroke teams
        one_hot = enc.fit_transform(outcome_data[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
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
        prediction = np.round(self.thrombolysis_choice_benchmark * 100, 0)
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

        def set_up_patient(patient, fields):
            p = patient[fields]
            enc = OneHotEncoder(categories=[self.stroke_teams])
            one_hot = enc.fit_transform(p[['stroke_team']]).toarray()
            one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
            p = pd.concat([p, one_hot], axis=1)
            p.drop('stroke_team', axis=1, inplace=True)
            return p
            
        def predict(choice_models, p):
            a = []
            for i in range(len(choice_models)):
                model = choice_models[i]
                a.append(model.predict_proba(p)[:, 1])
            return a

        def calculate_mean_std_ci(arr, n):
            """Calculate stats."""
            if len(arr.shape) > 1:
                # 2D array.
                m = np.mean(arr, axis=0)
                s = np.std(arr, axis=0)
            else:
                # 1D array.
                m = np.mean(arr)
                s = np.std(arr)
            sem = s / np.sqrt(n)
            ci = sem * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
            return m, s, ci

        patient = pd.DataFrame(patient_data, index=[0])

        # Get thrombolysis choice prediction
        fields = self.thrombolysis_choice_fields.copy()
        fields.remove('thrombolysis')
        patient_choice = set_up_patient(patient, fields)
        thrombolysis_predictions = predict(self.choice_models, patient_choice)
        # Convert to 1D list:
        thrombolysis_predictions = np.array(thrombolysis_predictions).flatten()
        # Stats:
        key = 'thrombolysis_prediction'
        m, s, c = calculate_mean_std_ci(thrombolysis_predictions,
                                        len(thrombolysis_predictions))
        setattr(self, key, m)
        setattr(self, f'{key}_std', s)
        setattr(self, f'{key}_ci', c)
        
        # Get benchmark thrombolysis predictions
        benchmark_predictions = []
        for benchmark_hosp in self.benchmark_hospitals:
            p = patient_choice.copy()
            # Change one-hot encoding
            current_team = patient_data['stroke_team']
            p[f'{current_team}'] = False
            p[f'{benchmark_hosp}'] = True
            # Get predictions
            thrombolysis_predictions = predict(self.choice_models, p)
            # Reset hospital
            p[f'{benchmark_hosp}'] = False
            # Get mean prediction
            benchmark_prediction = np.mean(np.array(thrombolysis_predictions))
            benchmark_predictions.append(benchmark_prediction)
        # Stats:
        key = 'thrombolysis_choice_benchmark'
        m, s, c = calculate_mean_std_ci(np.array(benchmark_predictions),
                                        len(benchmark_predictions))
        setattr(self, key, m)
        setattr(self, f'{key}_std', s)
        setattr(self, f'{key}_ci', c)

        # Get thrombolysis outcome prediction
        improvement = []
        fields = self.thrombolysis_outcome_fields.copy()
        fields.remove('discharge_disability')
        p_treated = set_up_patient(patient, fields)
        p_untreated = p_treated.copy()
        p_untreated['onset_to_thrombolysis'] = 99999

        n_models = len(self.outcome_models)
        
        dict_dists = {'untreated': {}, 'treated': {}}
        for d in dict_dists.keys():
            for k in ['dist', 'weighted_mrs', 'less_3', 'more_4']:
                dict_dists[d][k] = []
        for i in range(n_models):
            for t in dict_dists.keys():
                # Get untreated and treated distributions
                p_arr = p_untreated if t == 'untreated' else p_treated
                dist = self.outcome_models[i].predict_proba(p_arr).flatten()
                dict_dists[t]['dist'].append(dist)
                # Get weighted average of mRS scores
                weighted_dist = np.sum(dist * np.arange(7))
                dict_dists[t]['weighted_mrs'].append(weighted_dist)
                # Get untreated and treated distributions for mRS <3
                dict_dists[t]['less_3'].append(np.sum(dist[:3]))
                # Get untreated and treated distributions for mRS >4
                dict_dists[t]['more_4'].append(np.sum(dist[5:]))
            improvement.append(np.array(dict_dists['untreated']['weighted_mrs'][i]) -
                               np.array(dict_dists['treated']['weighted_mrs'][i]))

        # Calculate and store the mean, std and CI of the following 
        # arrays using attribute names from the dict keys:
        dict_arrays = {
            'untreated_dist': np.array(dict_dists['untreated']['dist']),
            'treated_dist': np.array(dict_dists['treated']['dist']),
            'untreated_less_3': np.array(dict_dists['untreated']['less_3']),
            'treated_less_3': np.array(dict_dists['treated']['less_3']),
            'untreated_more_4': np.array(dict_dists['untreated']['more_4']),
            'treated_more_4': np.array(dict_dists['treated']['more_4']),
            'untreated_weighted_mrs': (
                np.array(dict_dists['untreated']['weighted_mrs'])),
            'treated_weighted_mrs': (
                np.array(dict_dists['treated']['weighted_mrs'])),
            'improvement': np.array(improvement),
        }
        for key, arr in dict_arrays.items():
            m, s, c = calculate_mean_std_ci(arr, n_models)
            setattr(self, key, m)
            setattr(self, f'{key}_std', s)
            setattr(self, f'{key}_ci', c)
        # Second round now that some bits have been calculated:
        dict_arrays = {
            'change_in_less_3': (np.array(self.treated_less_3) -
                                 np.array(self.untreated_less_3)),
            'change_in_more_4': (np.array(self.treated_more_4) -
                                 np.array(self.untreated_more_4)),
        }
        for key, arr in dict_arrays.items():
            m, s, c = calculate_mean_std_ci(arr, n_models)
            setattr(self, key, m)
            setattr(self, f'{key}_std', s)
            setattr(self, f'{key}_ci', c)
        
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
