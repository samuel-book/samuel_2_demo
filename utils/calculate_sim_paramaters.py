import numpy as np
import pandas as pd


class CalculateSimParameters:

    """
    Summarises patient pathways processes and speeds for the pathway simulation model.

    For each stroke team records:

    * *thrombolysis_rate*: The proportion of all arrivals receiving thrombolysis
    * *admissions*: The number of admissions in the input data
    * *age_80_plus*: The proportion of patients aged 80+ (of those arriving within 4 hrs of known stroke onset)
    * *onset_known*: The proportion of patients with known onset time
    * *known_arrival_within_4hrs*: The proportion of patients arriving within 4 hours of known onset time
    * *onset_arrival_mins_mu*: Of those arriving in 4 hrs, the mean of log (ln) onset to arrival time
    * *onset_arrival_mins_sigma*: As above, but standard deviation
    * *scan_within_4_hrs*: Of those arriving in 4 hrs, the proportion of patients scanned within 4 hrs of arrival
    * *arrival_scan_arrival_mins_mu*: Of those arriving in 4 hrs and scanned within 4 hrs arrival, the mean of log (ln) of arrival to scan time
    * *arrival_scan_arrival_mins_sigma*: As above, but standard deviation
    * *onset_scan_4_hrs*: The proportion with onset-to-scan within 4 hrs
    * *eligible*: The proportion of patients with onset-to-scan within 4 hrs who receive thrombolysis
    * *scan_needle_mins_mu*: For those receiving thrombolysis, the mean log (ln) scan-to-needle time
* *scan_needle_mins_sigma*: A above, but standard deviation

    """

    def __init__(self, data_path, limit_to_ambo=False):
        """
        * Creates the CalculateSimParameters object and loads patient level data.
        * Required data is `patient_pathway_data.csv`
        * Limits to ambulance arrivals if required (default is False)
        * Removes stroke teams with fewer than 100 admissions in the input data
        """

        self.data_path = data_path

        # Load full data
        self.full_data = pd.read_csv(f'{self.data_path}/patient_pathway_data.csv', low_memory=False)

        # Limit to ambulance arrivals if required
        if limit_to_ambo:
            mask = self.full_data['arrive_by_ambulance'] == 1
            self.full_data = self.full_data[mask]

        # Remove stroke_team with fewer than 100 patients
        self.full_data = self.full_data.groupby('stroke_team').filter(lambda x: len(x) >= 100) 

    
    def calculate_parameters_for_pathway_simulation(self):
        """
        Calculate parameters for pathway simulation
        """

        # Set up results lists
        stroke_team = []
        admissions = []
        age_80_plus = []
        onset_known = []
        known_arrival_within_4hrs = []
        onset_arrival_mins_mu = []
        onset_arrival_mins_sigma = []
        scan_within_4_hrs = []
        arrival_scan_arrival_mins_mu = []
        arrival_scan_arrival_mins_sigma = []
        onset_scan_4_hrs = []
        scan_needle_mins_mu = []
        scan_needle_mins_sigma = []
        thrombolysis_rate = []
        eligible = []

        # Split data by stroke team
        groups = self.full_data.groupby('stroke_team')
        group_count = 0
        for index, group_df in groups: # each group has an index + dataframe of data
            group_count += 1

            # Record stroke team
            stroke_team.append(index)

            # Record  admission numbers
            admissions.append(group_df.shape[0])

            # Get thrombolysis rate
            thrombolysis_rate.append(group_df['thrombolysis'].mean())

            # Record onset known proportion and remove rest (unknown onset time)
            onset_known.append(group_df['onset_known'].mean())
            group_df = group_df[group_df['onset_known'] == 1]

            # Record onset-to-arrival <4hrs and remove rest (arrivals more than 4 hrs after stroke onset)
            mask = group_df['onset_to_arrival_time'] <= 240
            known_arrival_within_4hrs.append(mask.mean())
            group_df = group_df[mask]
            
            # Get proportion 80+ (of those arriving within 4 hours)
            over_80 = group_df['age'] >= 80
            age_80_plus.append(over_80.mean())

            # Log (ln) mean/sd of onset to arrival (remove any onset to arrival time of < 0)
            mask = group_df['onset_to_arrival_time'] > 0
            group_df = group_df[mask]
            ln_onset_to_arrival = np.log(group_df['onset_to_arrival_time'])
            onset_arrival_mins_mu.append(ln_onset_to_arrival.mean())
            onset_arrival_mins_sigma.append(ln_onset_to_arrival.std())

            # Remove any with arrival-to-scan time of <= 0
            mask = group_df['arrival_to_scan_time'] > 0
            group_df = group_df[mask]

            # Record proportion arrival-to-scan within 4 hours of arrival (and remove the rest)
            mask = group_df['arrival_to_scan_time'] <= 240
            scan_within_4_hrs.append(mask.mean())
            group_df = group_df[mask]

            # Log mean/sd of arrival to scan
            ln_arrival_to_scan = np.log(group_df['arrival_to_scan_time'])
            arrival_scan_arrival_mins_mu.append(ln_arrival_to_scan.mean())
            arrival_scan_arrival_mins_sigma.append(ln_arrival_to_scan.std())

            # Get proportion of patients with onset-to-scan of <=4hrs, and remove rest
            mask = (group_df['onset_to_arrival_time'] + 
                    group_df['arrival_to_scan_time']) <= 240
            onset_scan_4_hrs.append(mask.mean())
            group_df = group_df[mask]

            # Thrombolysis given (to remaining patients)
            eligible.append(group_df['thrombolysis'].mean())

            # Scan to needle
            mask = group_df['scan_to_thrombolysis_time'] > 0
            scan_to_needle = group_df['scan_to_thrombolysis_time'][mask]
            ln_scan_to_needle = np.log(scan_to_needle)
            scan_needle_mins_mu.append(ln_scan_to_needle.mean())
            scan_needle_mins_sigma.append(ln_scan_to_needle.std())

        # Store in DataFrame
        df = pd.DataFrame()
        df['stroke_team'] = stroke_team
        df['thrombolysis_rate'] = thrombolysis_rate
        df['admissions'] = admissions
        df['80_plus'] = age_80_plus
        df['onset_known'] = onset_known
        df['known_arrival_within_4hrs'] = known_arrival_within_4hrs
        df['onset_arrival_mins_mu'] = onset_arrival_mins_mu
        df['onset_arrival_mins_sigma'] = onset_arrival_mins_sigma
        df['scan_within_4_hrs'] = scan_within_4_hrs
        df['arrival_scan_arrival_mins_mu'] = arrival_scan_arrival_mins_mu
        df['arrival_scan_arrival_mins_sigma'] = arrival_scan_arrival_mins_sigma
        df['onset_scan_4_hrs'] = onset_scan_4_hrs
        df['eligable'] = eligible
        df['scan_needle_mins_mu'] = scan_needle_mins_mu
        df['scan_needle_mins_sigma'] = scan_needle_mins_sigma

        # Sort df by stroke_team
        df.sort_values(by=['stroke_team'], inplace=True)

        # Save to csv
        self.pathway_simulation_parameters = df
        self.pathway_simulation_parameters.to_csv(
            f'{self.data_path}/data_for_sim.csv', index=False)
