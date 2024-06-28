# SAMueL-2 Demo

Replication of Stroke Audit Machine Learning with artificial patient data.


## Run on Binder

You may run the notebooks on Binder. If it hasn't been used for a while it make take about 5 mins to initialise.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/samuel-book/samuel_2_demo/main)

## What is here?

These examples replicate key work from the SAMueL (Stroke Audit Machine Learning) project. Examples cover.

1) Choice of use of thrombolysis at each stroke team.

2) The effect of three possible changes to stroke pathways for each team:
    * Change pathway speed to 30 minutes arrival-to-thrombolysis for 95% of patients (15 min arrival-to-scan, 15 min scan-to-thrombolysis). We assume 5% of patients will be atypical and will not receive a scan within time for thrombolysis.
    * Change the proportion of patients with known onset time to the national upper quartile across stroke teams, if currently lower than that.
    * For those scanned in time to thrombolyse, apply a *benchmark* thrombolysis rate. This is the expected use of thrombolysis in the stroke team's own patient populations if decisions for each patient were made by a majority vote of 25 *benchmark* stroke teams which are the stroke teams with the highest willingness to use thrombolysis (predicted to have the highest use of thrombolysis if all stroke teams saw the same patients). *Benchmark teams* are predicted to have better outcomes (add reference when available).
    * A combination of above.

3) Predicted outcomes for patients with and without thrombolysis.

Both sets of models are for patients arriving within 4 hours of known stroke onset (onset time may be known precisely, or may have been estimated).

## Artificial patient data

### Machine Learning

Artificial patient data is created by sampling feature values (with replacement, and rounding) independently from patients attending 119 different stroke teams. Sampling is performed for each stroke team. These data do not maintain covariances in the original data (except that stroke severity is sampled separately for ischaemic and non-ischaemic stroke patients). These artificial patients are intended to demonstrate our machine learning models, and are not suitable for any clinical research into stroke. Each artificial patient has the following features.

* Stroke team (anonymised)
* Infarction (Y/N)
* Age (5 year age bands) - with ages censored below 35 or above 95
* Disability prior to stroke (modified Rankin Scale, mRS)
* Onset-to-arrival time (minutes, rounded to closest 5 mins)
* Arrival-to-scan time (minutes, rounded to closest 5 mins)
* Scan-to-thrombolysis time (minutes, if appropriate, rounded to closest 5 mins)
* Onset-to-thrombolysis time (calculated, if appropriate)
* Onset time known precisely (Y/N)
* Onset during sleep (all are then labelled as having imprecise onset times)
* Stroke Severity (NIHSS)
    Stroke severity is sampled separately for ischaemic and non-ischaemic strokes
* Atrial fibrillation coagulant (all also given a diagnosis of atrial fibrillation)

Stroke teams have been anonymised, and all stroke teams have 500 artificial patients generated.

Patient data was passed to models to predict probabilities of patients receiving thrombolysis, and then passed to a model to predict disability (mRS) on discharge from inpatient care. Use of thrombolysis and outcomes were sampled from distributions based on probabilities, and these outputs added to the artificial patients. While the artificial data does not maintain covariance of features, the labelling with use of thrombolysis and outcome will maintain interactions between features.

## Data for pathway model

The pathway sim data contains pathway summary information (mean and standard deviation) for 119 stroke teams. Admission numbers are set to 500/year for all teams. The team identity does not match the artificial patient data; both sets of data use arbitrary, and different, stroke team identification.

## Environment

A `envirnment.yml` file is provided in the `binder` folder. A summary of the packages used is:

* Python 3.10
* Matplotlib 3.8
* NumPy 1.26
* Pandas 2.1
* SHAP 0.43
* XGBoost 2.0

