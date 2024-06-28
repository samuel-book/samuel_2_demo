import numpy as np
import pandas as pd


class CalculateSimParameters:

    """
    Loads data ready for models.

    Attributes:

    full_data:
        Pandas dataframe of full SSNAP data (Cleaned)


    Methods:

    """

    def __init__(self, limit_to_ambo=False):
        """
        Creates the data load object
        """


        # Load full data
        self.full_data = pd.read_csv('./data/artificial_patient_pathway_data.csv', low_memory=False)

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

            # Record onset known proportion and remove rest
            onset_known.append(group_df['onset_known'].mean())
            group_df = group_df[group_df['onset_known'] == 1]

            # Record onset <4 hours and remove rest
            mask = group_df['onset_to_arrival_time'] <= 240
            known_arrival_within_4hrs.append(mask.mean())
            group_df = group_df[mask]
            
            # Calc proportion 80+ (of those arriving within 4 hours)
            over_80 = group_df['age'] >= 80
            age_80_plus.append(over_80.mean())

            # Log mean/sd of onset to arrival
            # Remove any with onset to arrival time of< 0
            mask = group_df['onset_to_arrival_time'] > 0
            group_df = group_df[mask]
            ln_onset_to_arrival = np.log(group_df['onset_to_arrival_time'])
            onset_arrival_mins_mu.append(ln_onset_to_arrival.mean())
            onset_arrival_mins_sigma.append(ln_onset_to_arrival.std())

            # Record scan within 4 hours of arrival (and remove the rest)
            # Remove any with arrival to scan time of < 0
            mask = group_df['arrival_to_scan_time'] > 0
            group_df = group_df[mask]
            mask = group_df['arrival_to_scan_time'] <= 240
            scan_within_4_hrs.append(mask.mean())
            group_df = group_df[mask]

            # Log mean/sd of arrival to scan
            ln_arrival_to_scan = np.log(group_df['arrival_to_scan_time'])
            arrival_scan_arrival_mins_mu.append(ln_arrival_to_scan.mean())
            arrival_scan_arrival_mins_sigma.append(ln_arrival_to_scan.std())

            # Record onset to scan in 4 hours and remove rest
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
            './data/data_for_sim.csv', index=False)
