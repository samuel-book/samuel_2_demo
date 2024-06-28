import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats 

class Pathway:

    """
    Stroke pathway simulation
    """
    def __init__(self, trials=30):
        # Load base case data
        self.hospital_performance_original = pd.read_csv('./data/data_for_sim.csv')

        # Set number of trials
        self.trials = trials

        # Load benchmark data
        self.benchmark_thrombolysis = pd.read_csv(
            './output/thrombolysis_rates.csv', index_col=0)

#       
    def model_ssnap_pathway_scenarios(self, hospital_performance):
        """
        Model of stroke pathway.
        
        Each scenario mimics 100 years of a stroke pathway. Patient times through 
        the pathway are sampled from distributions passed to the model using NumPy.
        
        Array columns:
        0: Patient aged 80+
        1: Allowable onset to needle time (may depend on age)
        2: Onset time known (boolean)
        3: Onset to arrival is less than 4 hours (boolean)
        4: Onset known and onset to arrival is less than 4 hours (boolean)
        5: Onset to arrival minutes
        6: Arrival to scan is less than 4 hours
        7: Arrival to scan minutes
        8: Minutes left to thrombolyse
        9: Onset time known and time left to thrombolyse
        10: Proportion ischaemic stroke (if they are filtered at this stage)
        11: Assign eligible for thrombolysis (for those scanned within 4 hrs of onset)
        12: Thrombolysis planned (scanned within time and eligible)
        13: Scan to needle time
        14: Clip onset to thrombolysis time to maximum allowed onset-to-thrombolysis
        15: Set baseline probability of good outcome based on age group
        16: Convert baseline probability good outcome to odds
        17: Calculate odds ratio of good outcome based on time to thrombolysis
        18: Patient odds of good outcome if given thrombolysis
        19: Patient probability of good outcome if given thrombolysis
        20: Clip patient probability of good outcome to minimum of zero
        21: Individual patient good outcome if given thrombolysis (boolean)*
        21: Individual patient good outcome if not given thrombolysis (boolean)*
        
        *Net population outcome is calculated here by summing probabilities of good
        outcome for all patients, rather than using individual outcomes. These columns
        are added for potential future use.
    
        """

        # Set up allowed time for thrombolysis (for under 80 and 80+)
        allowed_onset_to_needle = (270, 270)
        # Add allowed over-run 
        allowed_overrun_for_slow_scan_to_needle = 15
        # Set proportion of good outcomes for under 80 and 80+)
        good_outcome_base = (0.3499, 0.1318)

        # Set general model parameters
        scenario_counter = 0
        trials = 100
        
        # Set up dataframes
        
        results_columns = [
            'Baseline_good_outcomes_(median)',
            'Baseline_good_outcomes_per_1000_patients_(low_5%)',
            'Baseline_good_outcomes_per_1000_patients_(high_95%)',
            'Baseline_good_outcomes_per_1000_patients_(mean)',
            'Baseline_good_outcomes_per_1000_patients_(stdev)',
            'Baseline_good_outcomes_per_1000_patients_(95ci)',
            'Percent_Thrombolysis_(median%)',
            'Percent_Thrombolysis_(low_5%)',
            'Percent_Thrombolysis_(high_95%)',
            'Percent_Thrombolysis_(mean)',
            'Percent_Thrombolysis_(stdev)',
            'Percent_Thrombolysis_(95ci)',
            'Additional_good_outcomes_per_1000_patients_(median)',
            'Additional_good_outcomes_per_1000_patients_(low_5%)',
            'Additional_good_outcomes_per_1000_patients_(high_95%)',
            'Additional_good_outcomes_per_1000_patients_(mean)',
            'Additional_good_outcomes_per_1000_patients_(stdev)',
            'Additional_good_outcomes_per_1000_patients_(95ci)',
            'Onset_to_needle_(mean)']
        
        results_df = pd.DataFrame(columns=results_columns)
        
        # trial dataframe is set up each scenario, but define column names here
        # Rx = proportion given thrombolysis
        trial_columns = ['Baseline_good_outcomes',
                        'Rx',
                        'Additional_good_outcomes',
                        'onset_to_needle']
    
    
        # Iterate through hospitals
        for scenario_counter, run_data in hospital_performance.iterrows():

            # Skip if admissions less than 100
            if run_data['admissions'] < 100:
                continue
        
            # Set up trial results dataframe
            trial_df = pd.DataFrame(columns=trial_columns)
        
            for trial in range(self.trials):
                # %Set up numpy table
                patient_array = []
                patients_per_run = int(run_data['admissions'])
                patient_array = np.zeros((patients_per_run, 23))
        
                patient_array[:, 0] = \
                    np.random.binomial(1, run_data['80_plus'], patients_per_run)
        
                # Assign allowable onset to needle (for under 80 and 80+)
                patient_array[patient_array[:, 0] == 0, 1] = \
                    allowed_onset_to_needle[0]
                patient_array[patient_array[:, 0] == 1, 1] = \
                    allowed_onset_to_needle[1]
        
                # Assign onset time known
                patient_array[:, 2] = (np.random.binomial(
                    1, run_data['onset_known'], patients_per_run) == 1)
        
                # Assign onset to arrival is less than 4 hours
                patient_array[:, 3] = (
                    np.random.binomial(1, run_data['known_arrival_within_4hrs'], 
                    patients_per_run))
        
                # Onset known and is within 4 hours
                patient_array[:, 4] = patient_array[:, 2] * patient_array[:, 3]
        
                # Assign onset to arrival time (natural log normal distribution) 
                mu = run_data['onset_arrival_mins_mu']
                sigma = run_data['onset_arrival_mins_sigma'] 
                patient_array[:, 5] = np.random.lognormal(
                    mu, sigma, patients_per_run)
        
                # Assign arrival to scan is less than 4 hours
                patient_array[:, 6] = (
                    np.random.binomial(1, run_data['scan_within_4_hrs'],
                    patients_per_run))
        
                # Assign arrival to scan time (natural log normal distribution) 
                mu = run_data['arrival_scan_arrival_mins_mu']
                sigma = run_data['arrival_scan_arrival_mins_sigma']
                patient_array[:, 7] = np.random.lognormal(
                    mu, sigma, patients_per_run)
        
                # Minutes left to thrombolyse after scan
                patient_array[:, 8] = patient_array[:, 1] - \
                        (patient_array[:, 5] + patient_array[:, 7])
        
                # Onset time known, scan in 4 hours and 15 min ime left to thrombolyse
                # (1 to proceed, 0 not to proceed)
                patient_array[:, 9] = (patient_array[:, 6] * patient_array[:, 4] * 
                    (patient_array[:, 8] >= 15))
                
                # Ischaemic_stroke 
                # This is not used here - dealt with in 'eligble'. Set to 1.
                prop_ischaemic = 1 # run_data['ischaemic_stroke']
                patient_array[:, 10] = np.random.binomial(
                    1, prop_ischaemic, patients_per_run)
        
                # Eligable for thrombolysis (proportion of ischaemic patients  
                # eligable for thrombolysis when scanned within 4 hrs )
                patient_array[:, 11] = (
                    np.random.binomial(1, run_data['eligable'], patients_per_run))
        
                # Thrombolysis planned (checks this is within thrombolysys time, & 
                # patient considerd eligable for thrombolysis if scanned in time
                patient_array[:, 12] = (patient_array[:, 9] * patient_array[:, 10] *
                    patient_array[:, 11])

                # scan to needle
                mu = run_data['scan_needle_mins_mu']
                sigma = run_data['scan_needle_mins_sigma']
                patient_array[:, 13] = np.random.lognormal(
                    mu, sigma, patients_per_run)
        
                # Onset to needle 
                patient_array[:, 14] = \
                    patient_array[:, 5] + patient_array[:, 7] + patient_array[:, 13]
                
                # Clip to 4.5 hrs + given allowance max
                patient_array[:, 14] = np.clip(patient_array[:, 14], 0, 270 + 
                    allowed_overrun_for_slow_scan_to_needle)
        
                # Set baseline probability good outcome (based on age group)
                patient_array[patient_array[:, 0] == 0, 15] = good_outcome_base[0]
                patient_array[patient_array[:, 0] == 1, 15] = good_outcome_base[1]
        
                # Convert baseline probability to baseline odds
                patient_array[:, 16] = (patient_array[:, 15] /
                    (1 - patient_array[:, 15]))
        
                # Calculate odds ratio based on time to treatment
                patient_array[:, 17] = 10 ** (0.326956 + 
                    (-0.00086211 * patient_array[:, 14]))
        
                # Adjust odds of good outcome
                patient_array[:, 18] = patient_array[:, 16] * patient_array[:, 17]
        
                # Convert odds back to probability
                patient_array[:, 19] = (patient_array[:, 18] / 
                    (1 + patient_array[:, 18]))
        
                # Improved probability of good outcome (calc changed probability 
                # then multiply by whether thrombolysis given)
                x = ((patient_array[:, 19] - patient_array[:, 15]) * patient_array[:, 12])
                
                y = np.zeros(patients_per_run)
                
                # remove any negative probabilities calculated
                # (can occur if long treatment windows set)
                patient_array[:, 20] = np.amax([x, y], axis=0)
        
                # Individual good ouctome due to thrombolysis 
                # This is not currently used in the analysis
                patient_array[:, 21] = np.random.binomial(
                    1, patient_array[:, 20], patients_per_run)
        
                # Individual outcomes if no treatment given 
                patient_array[:, 22] = np.random.binomial(
                    1, patient_array[:, 15], patients_per_run)
        
                # Calculate overall thrombolysis rate
                thrmobolysis_percent = patient_array[:, 12].mean() * 100
        
                # Baseline good outcomes per 1000 patients
                baseline_good_outcomes_per_1000_patients = (
                    (patient_array[:, 22].sum() / patients_per_run) * 1000)
        
                # Calculate overall expected extra good outcomes
                additional_good_outcomes_per_1000_patients = (
                    ((patient_array[:, 20].sum() / patients_per_run) * 1000))
                
                # Extract times for thrombolysis
                thrombolysis_results = pd.DataFrame()
                mask = patient_array[:,12] == 1
                thrombolysis_results['onset_to_arrival'] = patient_array[:,5]
                thrombolysis_results['arrival_to_scan'] = patient_array[:,7]
                thrombolysis_results['scan_to_needle'] = patient_array[:,13]
                thrombolysis_results['onset_to_needle'] = patient_array[:,14]
                
                onset_to_needle = \
                        thrombolysis_results['onset_to_needle'][mask].mean()
                
            
                # Save scenario results to dataframe
                result = [baseline_good_outcomes_per_1000_patients, 
                        thrmobolysis_percent,
                        additional_good_outcomes_per_1000_patients,
                        onset_to_needle]
                trial_df.loc[trial] = result
                
        
            trial_result = ([
                trial_df['Baseline_good_outcomes'].median(),
                trial_df['Baseline_good_outcomes'].quantile(0.05),
                trial_df['Baseline_good_outcomes'].quantile(0.95),
                trial_df['Baseline_good_outcomes'].mean(),
                trial_df['Baseline_good_outcomes'].std(),
                (trial_df['Baseline_good_outcomes'].mean() - 
                    stats.norm.interval(0.95, loc=trial_df['Baseline_good_outcomes'].mean(),
                    scale=trial_df['Baseline_good_outcomes'].std() / sqrt(trials))[0]),
                trial_df['Rx'].median(),
                trial_df['Rx'].quantile(0.05),
                trial_df['Rx'].quantile(0.95),
                trial_df['Rx'].mean(),
                trial_df['Rx'].std(),
                (trial_df['Rx'].mean() - stats.norm.interval(
                    0.95, loc=trial_df['Rx'].mean(),
                    scale=trial_df['Rx'].std() / sqrt(trials))[0]),
                trial_df['Additional_good_outcomes'].median(),
                trial_df['Additional_good_outcomes'].quantile(0.05),
                trial_df['Additional_good_outcomes'].quantile(0.95),
                trial_df['Additional_good_outcomes'].mean(),
                trial_df['Additional_good_outcomes'].std(),
                (trial_df['Additional_good_outcomes'].mean() - 
                    stats.norm.interval(0.95, loc=trial_df['Additional_good_outcomes'].mean(),
                    scale=trial_df['Additional_good_outcomes'].std() / sqrt(trials))[0]),
                trial_df['onset_to_needle'].mean()
                ])
            # add scenario results to results dataframe
            results_df.loc[run_data['stroke_team']] = trial_result
       
        # round all results to 2 decimal places and return    
        results_df = results_df.round(2)

        return results_df
    

    def plot_summary_results(self):

        """Plot and save overall simulation results"""

        fig = plt.figure(figsize=(10,7))

        rows = ['base',
             'onset',
             'speed',
             'benchmark',
             'speed_onset',
             'speed_benchmark',
             'onset_benchmark',
             'speed_onset_benchmark']
        
        # reorder rows to above
        self.summary_sim_results = self.summary_sim_results.reindex(rows)

        x = list(self.summary_sim_results.index)


        # Replace all _ in x with + for plotting
        x = [i.replace('_', '+') for i in x]

        ax1 = fig.add_subplot(121)        
        y1 = self.summary_sim_results['Percent_Thrombolysis'].values
        ax1.bar(x,y1)
        ax1.set_ylim(0,20)
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0,22,2))
        ax1.set_title('Thrombolysis use (%)')
        ax1.set_ylabel('Thrombolysis use (%)')
        ax1.set_xlabel('Scenario')
        ax1.grid(axis = 'y')

        ax2 = fig.add_subplot(122)
        y1 = self.summary_sim_results['Additional_good_outcomes_per_1000_patients'].values
        ax2.bar(x,y1, color='r')
        ax2.set_ylim(0,20)
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0,22,2))
        ax2.set_title('Additional good outcomes\nper 1,000 admissions')
        ax2.set_ylabel('Additional good outcomes\nper 1,000 admissions')
        ax2.set_xlabel('Scenario')
        ax2.grid(axis = 'y')

        plt.tight_layout(pad=2)

        plt.savefig('./output/sim_results_summary.jpg', dpi=300)
        plt.close()

    def run(self):
        """Model scenarios"""

        # Run base case scenario
        self.sim_results = self.model_ssnap_pathway_scenarios(self.hospital_performance_original)
        self.sim_results['scenario'] = 'base'

        ############################################## 
        # SPEED (30 minutes arrival to thrombolysis) #
        ##############################################

        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()
        hospital_performance['scan_within_4_hrs'] = 0.95
        hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
        hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
        hospital_performance['scan_needle_mins_mu'] = np.log(15)
        hospital_performance['scan_needle_mins_sigma'] = 0
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'speed'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        ##########################################################
        # KNOWN ONSET (SET to UPPER QUARTILE if currently lower) #
        ##########################################################

        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()

        onset_known = self.hospital_performance_original['onset_known']
        onset_known_upper_q = np.percentile(onset_known, 75)
        adjusted_onset_known = []
        for val in onset_known:
            if val > onset_known_upper_q:
                adjusted_onset_known.append(val)
            else:
                adjusted_onset_known.append(onset_known_upper_q)
        hospital_performance['onset_known'] = adjusted_onset_known
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'onset'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        ################################
        # Benchmark thrombolysis rates #
        ################################

        # Merge in benchmark rates (to ensure order is correct)
        hospital_performance = self.hospital_performance_original.copy()
        hospital_performance = hospital_performance.merge(
            self.benchmark_thrombolysis, left_on='stroke_team', right_index=True, how='left')
        benchmark_adjustment = (hospital_performance['benchmark_decision'] / 
                                hospital_performance['thrombolysis'])
        hospital_performance['eligable'] *= benchmark_adjustment
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'benchmark'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        ###################
        # Onset and speed #
        ###################

        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()
        # Speed
        hospital_performance['scan_within_4_hrs'] = 0.95
        hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
        hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
        hospital_performance['scan_needle_mins_mu'] = np.log(15)
        hospital_performance['scan_needle_mins_sigma'] = 0
        # Onset known
        onset_known = self.hospital_performance_original['onset_known']
        onset_known_upper_q = np.percentile(onset_known, 75)
        adjusted_onset_known = []
        for val in onset_known:
            if val > onset_known_upper_q:
                adjusted_onset_known.append(val)
            else:
                adjusted_onset_known.append(onset_known_upper_q)
        hospital_performance['onset_known'] = adjusted_onset_known
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'speed_onset'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        #############################
        # Speed and benchmark rates #
        #############################

        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()
        # Speed
        hospital_performance['scan_within_4_hrs'] = 0.95
        hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
        hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
        hospital_performance['scan_needle_mins_mu'] = np.log(15)
        hospital_performance['scan_needle_mins_sigma'] = 0
        # Merge in benchmark rates (to ensure order is correct)
        hospital_performance = hospital_performance.merge(
            self.benchmark_thrombolysis, left_on='stroke_team', right_index=True, how='left')
        benchmark_adjustment = (hospital_performance['benchmark_decision'] / 
                                hospital_performance['thrombolysis'])
        hospital_performance['eligable'] *= benchmark_adjustment
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'speed_benchmark'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        #############################
        # Onset and benchmark rates #
        #############################

        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()
        # Onset known
        onset_known = self.hospital_performance_original['onset_known']
        onset_known_upper_q = np.percentile(onset_known, 75)
        adjusted_onset_known = []
        for val in onset_known:
            if val > onset_known_upper_q:
                adjusted_onset_known.append(val)
            else:
                adjusted_onset_known.append(onset_known_upper_q)
        hospital_performance['onset_known'] = adjusted_onset_known
        # Merge in benchmark rates (to ensure order is correct)
        hospital_performance = hospital_performance.merge(
            self.benchmark_thrombolysis, left_on='stroke_team', right_index=True, how='left')
        benchmark_adjustment = (hospital_performance['benchmark_decision'] / 
                                hospital_performance['thrombolysis'])
        hospital_performance['eligable'] *= benchmark_adjustment
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'onset_benchmark'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        #######
        # All #
        #######
        # Create scenarios
        hospital_performance = self.hospital_performance_original.copy()
        # Onset known
        onset_known = self.hospital_performance_original['onset_known']
        onset_known_upper_q = np.percentile(onset_known, 75)
        adjusted_onset_known = []
        for val in onset_known:
            if val > onset_known_upper_q:
                adjusted_onset_known.append(val)
            else:
                adjusted_onset_known.append(onset_known_upper_q)
        hospital_performance['onset_known'] = adjusted_onset_known
        # Merge in benchmark rates (to ensure order is correct)
        hospital_performance = hospital_performance.merge(
            self.benchmark_thrombolysis, left_on='stroke_team', right_index=True, how='left')
        benchmark_adjustment = (hospital_performance['benchmark_decision'] / 
                                hospital_performance['thrombolysis'])
        hospital_performance['eligable'] *= benchmark_adjustment
        # Speed
        hospital_performance['scan_within_4_hrs'] = 0.95
        hospital_performance['arrival_scan_arrival_mins_mu'] = np.log(15)
        hospital_performance['arrival_scan_arrival_mins_sigma'] = 0
        hospital_performance['scan_needle_mins_mu'] = np.log(15)
        hospital_performance['scan_needle_mins_sigma'] = 0
        # Get results
        results = self.model_ssnap_pathway_scenarios(hospital_performance)
        results['scenario'] = 'speed_onset_benchmark'
        # Add to results_all
        self.sim_results = pd.concat([self.sim_results, results], axis=0)

        # Merge in admission numbers
        self.sim_results = self.sim_results.merge(
            self.hospital_performance_original[['stroke_team', 'admissions']],
            left_index=True, right_on='stroke_team', how='left')
        
        # Average over stroke teams
        columns = ['scenario',
                   'Percent_Thrombolysis_(mean)', 
                   'Additional_good_outcomes_per_1000_patients_(mean)']
        df = self.sim_results[columns].groupby('scenario').mean()
        self.summary_sim_results = \
            df.rename(columns={'Percent_Thrombolysis_(mean)': 'Percent_Thrombolysis',
            'Additional_good_outcomes_per_1000_patients_(mean)': 'Additional_good_outcomes_per_1000_patients'}).round(2)
        
        # Plot
        self.plot_summary_results()

        # Save results
        self.sim_results.to_csv('./output/sim_results_all.csv', index=False)
        self.summary_sim_results.to_csv('./output/sim_results_summary.csv', index=True)



