# Probability, odds, and SHAP values (log odds shifts)

Many of us find it easiest to think of the chance of something occurring as a probability. For example, there might be a probability of 10% that it will rain today. That is the same as saying there will be one rainy day out of ten days for days with this given probability of rain.

In our stroke thrombolysis model, SHAP values tell us how knowing something particular about a patient (such as the patient *feature*, 'Is their stroke caused by a clot or a bleed?') adjusts our prediction of whether they will receive thrombolysis or not.

This is made a little more complicated for us because SHAP is usually reported as a *log odds shift*. It is useful for us to see how those relate to probabilities, and get a sense of how significant SHAP values in the range of 0.5 to 5 (or -0.5 to -5) are, as that is a common range of SHAP values that we will see in our models.

## Probability

We will take the example that SHAP reports that a model's base probability prediction, before the contribution of features is 0.25, or a 25% probability of receiving thrombolysis; that is 1 in 4 patients with this prediction would be expected to receive thrombolysis.

## Odds

*Probability* expresses the chance of something happening as the number of positive occurrences as a fraction of all occurrences (i.e. the number of patients receiving thrombolysis as a fraction of the total number of patients).

*Odds* express the chance of something happening as the ratio of the number of positive occurrences (i.e. receiving thrombolysis) to the number of negative occurrences (i.e. *not* receiving thrombolysis).

If we have probability prediction of 0.25 would receive thrombolysis, that would mean 1 in 4 of those patients receive thrombolysis. Expressed as odds, for every one patient that receives thrombolysis, three will not. The odds are expressed as 1:3 or 1/3. This may also be calculated as a decimal (1 divided by 3), 0.333.

Odds (O) and probability (P) may be converted with the following equations:

(1) O = P / (1 - P)

(2) P = O / (1 + O)

## SHAP values: log odds shifts

Here we will calculate the effect of SHAP values, and try and build some intuition on the size of effect SHAP values of 0.5 to 5 give (we will look at positive and negative SHAP values).

SHAP usually outputs the effect of a particular feature in how much it shifts the odds. For reasons we will not go into here, that shift (which is the 'SHAP value') is usually given in 'log odds' (the logarithm of the odds value). For the mathematically inclined, we use the natural log (*ln*).

Let's look at some SHAP values (log odds) and see how much they change the odds of receiving thrombolysis. 

First we'll look at the shift in odds the SHAP values give. This is calculated as *shift = exp(SHAP)*

| SHAP (log odds) | Shift in odds (multiply original odds) |
|-----------------|----------------------------------------|
| 0.5             | 1.65                                   |
| 1               | 2.72                                   |
| 2               | 7.39                                   |
| 3               | 20.1                                   |
| 4               | 54.6                                   |
| 5               | 148                                    |

### Positive SHAP values: examples of how SHAP change odds and probabilities

Now let us work through an example of starting with a known baseline *probability* (before we consider what we know about a particular patient feature), converting that to *odds*, applying a SHAP *log odds shift* for that particular feature, and converting back to *probability* after we have applied the influence of that feature.

Here are the effects of those shifts on our baseline probability of 0.25.

| Starting P | Starting O<br>(P / (1 - P)) | SHAP | Shift in odds<br>(exp(SHAP)) | Shifted O<br>(O * Shift) | Shifted P (%)<br>(O / (1 + O)) |
|------------|-----------------------------|------|------------------------------|--------------------------|--------------------------------|
| 0.25 (25%) | 0.333                       | 0.5  | 1.65                         | 0.550                    | 0.3547 (35.5%)                 |
| 0.25 (25%) | 0.333                       | 1    | 2.72                         | 0.907                    | 0.4754 (47.5%)                 |
| 0.25 (25%) | 0.333                       | 2    | 7.39                         | 2.46                     | 0.7112 (71.1%)                 |
| 0.25 (25%) | 0.333                       | 3    | 20.1                         | 6.70                     | 0.8700 (87.0%)                 |
| 0.25 (25%) | 0.333                       | 4    | 54.6                         | 18.2                     | 0.9479 (94.8%)                 |
| 0.25 (25%) | 0.333                       | 5    | 148                          | 49.5                     | 0.9802 (98.0%)                 |


So, for example, a SHAP value of 0.5 for one particular feature tells us that that particular feature in that patient shifts our expected probability of that patient receiving thrombolysis from 25% to 36%. A SHAP value of 5 for the same feature would shift the probability of that patient receiving thrombolysis up to 98%.

### Negative SHAP values: examples of how SHAP change odds and probabilities

If we have a negative SHAP value then odds are reduced (a SHAP of -1 will lead to the odds being divided by 2.72, which is the same as multiplying by 1/2.72, which is 0.3679):

| Starting P | Starting O<br>(P / (1 - P)) | SHAP | Shift in odds<br>(exp(SHAP)) | Shifted O<br>(O * Shift) | Shifted P (%)<br>(O / (1 + O)) |
|------------|-----------------------------|------|------------------------------|--------------------------|--------------------------------|
| 0.25 (25%) | 0.333                       | -0.5 | 0.6065                       | 0.2022                   | 0.1682 (16.8%)                 |
| 0.25 (25%) | 0.333                       | -1   | 0.3679                       | 0.1226                   | 0.1092 (10.9%)                 |
| 0.25 (25%) | 0.333                       | -2   | 0.1353                       | 0.0451                   | 0.0432 (4.32%)                 |
| 0.25 (25%) | 0.333                       | -3   | 0.0498                       | 0.0166                   | 0.0163 (1.63%)                 |
| 0.25 (25%) | 0.333                       | -4   | 0.0183                       | 0.0061                   | 0.0061 (0.61%)                 |
| 0.25 (25%) | 0.333                       | -5   | 0.0067                       | 0.0022                   | 0.0022 (0.22%)                 |

So, for example, a SHAP value of -0.5 for one particular feature tells us that that particular feature in that patient shifts our expected probability of that patient receiving thrombolysis from 25% to 17%. A SHAP value of 5 for the same feature would shift the probability of that patient receiving thrombolysis down to 2%.

## A worked example in predicting use of thrombolysis in stroke

The example below shows how six patient features change the model prediction of a patient receiving thrombolysis, from a baseline of 25% (with nothing known about the patient) to 95% after the contribution of six features. The relative contribution of each feature is made clear by its SHAP value. SHAP are additive in their effects; for example if one feature has a SHAP of +1.5, the influence of that feature on the model prediction would be exactly cancelled out by three other features with SHAP values of -0.5.

| Patient feature and value        | SHAP | Shift<br>(exp(SHAP)) | Odds<br>(Previous odds * Shift) | Probability<br>(O / (1 + O) |
|----------------------------------|------|----------------------|---------------------------------|-----------------------------|
| Base odds                        | N/A  | N/A                  | 0.33                            | 25%                         |
| Stroke type = infarction         | 1.8  | 6.05                 | 2                               | 67%                         |
| Stroke severity (NIHSS) = 20     | 1.5  | 4.482                | 8.95                            | 90%                         |
| Prior disability (mRS) = 3       | -0.7 | 0.497                | 4.44                            | 82%                         |
| Precise onset time = Yes         | 0.6  | 1.822                | 8.1                             | 89%                         |
| Arrival-to-scan time (mins) = 30 | 0.5  | 1.649                | 13.35                           | 93%                         |
| Use of AF anticoagulants = No    | 0.3  | 1.35                 | 18.02                           | 95%                         |

### Observations about SHAP values

We begin to get some intuition on SHAP values. A SHAP value of 0.5 (or -0.5) leads to a small, but still noticeable, change in probability. SHAP values of 5 or -5 have effectively pushed probabilities to one extreme or the other.

Everything we know about a patient (the patient 'features') may shift the predicted probability of receiving thrombolysis. Each patient feature will have its own SHAP value depending on the value of that feature. *Stroke severity*, for example, may be in the range 0-42, and the SHAP value will depend on the value of the feature. The final predicted probability of receiving thrombolysis will be the model baseline prediction (the predicted probability of receiving thrombolysis before anything is known about a patient) and the sum of the influences of each of the features.







