## Alzheimer’s Prediction – A Machine Learning Model Comparison

#### Rudy Tewelde
##### DSC 522 – Machine Learning II

# Introduction

Alzheimer’s Disease (AD) is a progressive disease of the brain that slowly causes impairment in memory and cognitive function. The exact cause is unknown, and no cure is available. The National Institutes of Health estimates that more than 5 million people in the United States have Alzheimer’s. Alzheimer’s is currently the sixth-leading cause of death in the US. Although young persons can and do get Alzheimer’s, the symptoms generally begin after age 60. The time from diagnosis to death can be as little as three years in persons over the age of 80.  There is a worldwide effort underway to find better ways to treat the disease, delay its onset, and prevent it from developing.

Brain Imaging via magnetic resonance imaging (MRI) is used to evaluate patients with suspected AD. Some studies have suggested that MRI features may predict the rate of decline in AD and may guide therapy in the future. However, assessing an individual’s current diagnosis can vary from one clinician to the next, or from day to day.  This study found high subjectivity and individual-level variability in cognitive assessments.

The primary goal is two-fold; one is to predict whether a patient has Alzheimer’s disease; the second is to identify individuals at risk of Alzheimer’s disease.
The aim is to develop sound models that may help clinicians catch Alzheimer’s early and predict risk factors associated with Alzheimer’s via machine learning by implementing classification algorithms for the analysis of clinical data and providing a prediction tool for the early diagnosis of the disease. The chosen methodology will be determined by fitting the data to multiple machine learning algorithms and selecting those that provide the best predictive capabilities.

# Data

The data was obtained from the Open Access Series of Imaging Studies (OASIS). OASIS is a project aimed at making MRI data sets of the brain freely available to the scientific community. OASIS is comprised of two data sets, one of longitudinal MRI Data and one of cross-sectional MRI data. These data sets are used in training multiple machine learning models to identify patients with mild to moderate dementia. Because Alzheimer’s is the most common cause of dementia, these data sets are chosen.  

* Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults: 

This set consists of a cross-sectional collection of 416 subjects aged 18 to 96. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. One hundred of the included subjects over the age of 60 have been clinically diagnosed with very mild to moderate Alzheimer’s disease. Additionally, a reliability data set includes 20 non-demented subjects imaged on a subsequent visit within 90 days of their initial session.

* Longitudinal MRI Data in Nondemented and Demented Older Adults:

This set consists of a longitudinal collection of 150 subjects aged 60 to 96. Each subject was scanned on two or more visits, separated by at least one year for a total of 373 imaging sessions. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. Seventy-two of the subjects were characterized as non-demented throughout the study. Sixty-four of the included subjects were characterized as demented at the time of their initial visits and remained so for subsequent scans, including 51 individuals with mild to moderate Alzheimer’s disease. Another 14 subjects were characterized as non-demented at the time of their initial visit and were subsequently characterized as demented at a later visit.

The 150 patients in the longitudinal study ranged in age from 60 to 96 years. Information about their age, socioeconomic status, and education was included, along with various anatomical factors. Each patient was subject to an MRI at least twice throughout the study. Their brain volume was measured, and the MMSE (Mini-Mental State Examination) score was recorded. Each subject was then classified as either demented or non-demented.

The participants in both studies include young, middle-aged, and older adults, both demented and non-demented. Ultimately, to maintain the models’ integrity, only the longitudinal data set was utilized for analysis due to several missing observations in the cross-sectional data.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/1.png)


EDUC: Education in years.

SES: Socio-economic status, assessed by Hollingshead Index of Social Position and categorized from 1 (highest) to 5 (lowest).

MMSE: Mini-Mental State Examination. Scores range from 0 to 30. The breakdown of the score ranges is below.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/2.png)
CDR: Clinical Dementia Rating is a 5-point scaling system used to determine cognitive impairment and overall function. The breakdown of the scores is shown below.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/3.png)

eTIV: Estimated Total Intracranial Volume. An important metric used to examine brain function. Studies show that brain volume and neurodegenerative diseases such as Alzheimer’s are linked. In this collection, eTIV was measured using mm3 sections from MRI scans.

nWBV: Normalized Whole-Brain Volume, expressed as a percent to quantify the degeneration in masked images of either gray or white matter. This measure of the brain’s volume is often used to assess atrophy of the brain tissue, as atrophy is one of the symptoms of Alzheimer’s.  

ASF: Atlas Scaling Factor. The normalization commonly used by functional data analysis as an automated solution to the widely encountered problem of correcting for head size variation in regional and whole-brain scans.

MR Delay: The delay time given before the actual image acquisition to optimize the difference in signals from the targeted tissues.

# Exploratory Data Analysis
The count of subjects across the Clinical Dementia Rating scores were: 206 with a score of zero, 123 with a score of 0.5, 41 subjects with score of one, and three subjects with score of two.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/4a.png)

The groups include 190 Non-demented, 146 Demented, and 37 Converted. Subjects labeled as converted had some form of cognitive impairment that may manifest into dementia but not always. These were later treated as Nondemented for modeling purposes.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/5.png)

The counts by gender included 213 female and 160 male.
![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/6.png)

The following graphs highlight the tendency of those labeled non-demented as having higher brain volume ratio than those labeled Demented. This is believed to be due to the diseases effect on the brain includes a shrinking of its tissue.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/7.png)

The following graph of the demented group by age shows a higher concentration of 70-80 years old in the demented patient group than those in the non-demented grouping. The extant literature purports that patients at these ages have lower survival rates. The dip and cross of the curves beginning after 80 and approaching 90 also suggests that.
![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/8.png)


# Data Pre-Processing
There were relatively few transformations necessary before the features of the data could be interpreted by the algorithms. The longitudinal data arrived relatively clean, although there were 19 values missing from SES and 2 from MMSE; both were imputed with the median values for their respective fields. With the data being composed of attributes with varying scales, both a standard scaler and a robust scaler were considered. The robust scaler was ultimately used, as it still centers the data but is more robust to outliers, of which there are many. In order to avoid leakage of information about parameters between the training and testing sets, pipelines were used to scale each set separately. Pipelines were also used in order to streamline the scaling, model fits, and grid searches for each model run.

# Feature Selection
Univariate correlations were plotted from the original set and its engineered variables to determine each variable’s relationship with all other variables. Two-way correlations were run between variables to measure correlations as well as the distribution of values such as those found in the example below.
![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/9.png)

Correlations were also computed for the entirety of the data (with the absence of CDR, hand, and visit) as shown below. The relationships between the engineered fields and the variables they were created from and the different measures of brain volume are as expected. Other notable interactions include age by normalized whole brain volume and between educ_age (the time since the last schooling) and nWBV, which may suggest that more recent education, or mental stimulation, has a positive impact on higher brain volume.

 ![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/10.png)

In order to determine the features most important for modeling, multiple feature importance methodologies are applied: Recursive Feature Elimination (RFE), feature importances based on tree models and permutation importance. For consistency, XGBoost was the model specified for each method used. 

Feature importances are automatically factored into tree-based models, but making sense of them is a different matter. For instance, the default _feature_importances_ command is limited to giving each importance as a percentage, adding up to one. That said, we can also show how many leaves and how much information was gained from each feature. For a gradient boosted (or extreme gb) model, the leaves used are misleading; gradient boosted models pay more attention to features confusing to it, and thus it is usually the features with fewer leaves that are more important to the model itself. The differences between these is shown below.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/11.png)
  
The figure on the above left shows how many leaves were used, while the one on the right shows information gained (calculated by the Gini coefficient).

Other ways to show features’ importance in a model include permutation importance as well as a method called SHaply Additive exPlanations (SHAP). Permutation importance essentially takes each field and permutates the values within, running the desired model again for each feature. The idea being that for noisier features, permutating the values within that field will have no real effect on the performance of the model and may even improve the output. However, for important features, permutating the values will drastically affect the performance. The permutation importance statement ranks each feature by how much the model decreased in performance when its values were shuffled, with only the features whose permutation resulted in an increase in performance marked in red.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/12.png)

Shown above, MMSE is marked as extremely important; if we randomly shuffle its values around, this tells us that the model would decrease in accuracy by almost 5%. Conversely, for our int_ses variable, it was found to be noisy and increased model performance by up to 2% when its values were shuffled.

SHAP is a compelling library for model explanation, using a game-theoretic approach to explain a model’s decision in contrast to a baseline value (what we would decide with no prior information based on the distribution of the classes). We can use SHAP values to explain partial dependence, individual predictions, and the effect of higher and lower values for each feature used in the model. As previously mentioned, to explain a prediction on dementia, it can help to see decisions for individual predictions, as shown below.

 ![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/13.png)


In the figure above, we see the base value; or the proportion of demented vs. non-demented patients. We also see the output value of 0.6, meaning a positive (and correct) prediction of dementia. For each feature in the model, we know the magnitude of its effect in the model’s decision, with a 27 for MMSE having the most significant contribution. We also see the nWBV score pushing the model towards a prediction of non-dementia, though not nearly as strongly as MMSE.

SHAP also allows for the use of partial dependence plots to investigate interactions between features and determining their high and low values. Another advantage of using SHAP over other methods is that we can look for trends in the interaction while intuitively assessing if the features correlate well in their contribution or not.

 ![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/14.png)

Above is an example of such a plot; we see that an increase in age means a general decrease in the probability of dementia, but we can also see very scattered colors for education. This suggests that age and education do not contribute in the same direction (an increase in one’s magnitude leads to a rise in the other’s). Finally, we can plot each feature’s importance for their high and low values in the model, which can tell us the range of contributions for each feature.
![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/15.png)

 
Above, the MMSE again has the most considerable total importance, with the high values of MMSE all lowering the probability of dementia, while the middle to low values all raise the probability. Also, for most features, there is a somewhat visible split between the contribution of the high and low values. 

# Results

After determining which features were most important, the data was fitted to multiple machine learning algorithms in order to assess which provided the best predictions of Alzheimer’s. A variety of popular algorithms were tried in addition to some niche models. Each model’s parameters were tuned using GridSearch. Nearly all the models performed at an accuracy level of 80% or higher. The results of each model is described below.


Logistic Regression
The logistic regression model performed with an accuracy score of 83.93%. From the confusion matrix, the model correctly classified 56 patients as non-demented. Ten non-demented patients were misclassified as demented. There were eight demented patients falsely classified as non-demented. Thirty-eight demented patients were  classified correctly as demented.
  
Multi-Layer Perceptron
The MLP model performed with an accuracy score of 81.25%. From the confusion matrix, we can see that 55 patients were correctly classified as non-demented. Eleven non-demented patients were misclassified as demented. There were ten demented patients falsely classified as non-demented. Lastly, there were 36 demented patients correctly classified as demented.  

K-Nearest Neighbors
The KNN model performed with an accuracy score of 80.36%. From the confusion matrix, we see that 59 patients were correctly classified as non-demented. Seven non-demented patients were misclassified as demented. There were 15 demented patients falsely classified as non-demented. Lastly, there were 31 demented patients correctly classified as demented.  

XGBoost
The XGBoost model performed best, with an accuracy score of 84.82%. From the confusion matrix, we can see that 57 patients were correctly classified as non-demented. Nine non-demented patients were misclassified as demented. There were eight demented patients falsely classified as non-demented. Lastly, there were 38 demented patients correctly classified as demented. 

Support Vector Machines
The SVM model performed identically as well as the LogReg model, with an accuracy score of 83.93%. From the confusion matrix, we can see that 56 patients were correctly classified as non-demented. Ten non-demented patients were misclassified as demented. There were eight demented patients falsely classified as non-demented. Lastly, there were 38 demented patients correctly classified as demented.  

Keras
The Keras classification model was the poorest performer, with an accuracy score of 78.57%. From the confusion matrix, we can see that 60 patients were correctly classified as non-demented. Six non-demented patients were misclassified as demented. There were 18 demented patients falsely classified as non-demented. Lastly, there were 28 demented patients correctly classified as demented.

![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/16.png)

 ![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/17.png)

# Limitations
Various models failed to interpret the same few points, and we suspect the presence of significant noise in our data. The main limitation to the modeling was the quantity of available data. As with most traditional machine learning algorithms, more data generally lends itself to better results. With a more substantial amount of training data, the myriad of techniques, algorithms, and complex architectures that can significantly surpass any human-like abilities for many of the specific classification tasks studied are available, could play a critical role in the future success of machine learning model’s accuracy on unseen data.

Alzheimer’s is a common neurodegenerative disease, with no known cure. Using a clinical MRI data set obtained from OASIS, the presence of dementia was predicted in patients with up to an 85% accuracy level using a variety of features. Utilizing feature importance packages, which specific factors would be most helpful in predicting Alzheimer’s was determined. The potency of feature combination in contributing to the accurate prediction of the disease was observed. With the use of pipelines and paying careful attention to leaky strategy and leaky features, a process was created that could be packaged for clinicians to replicate in the field. 

# Acknowledgments 
Data were provided by OASIS http://www.oasis-brains.org/#data

i.	OASIS: Cross-Sectional: Principal Investigators: D. Marcus, R, Buckner, J, Csernansky J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
ii.	OASIS: Longitudinal: Principal Investigators: D. Marcus, R, Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382
iii.	OASIS-3: Principal Investigators: T. Benzinger, D. Marcus, J. Morris; NIH P50AG00561, P30NS09857781, P01AG026276, P01AG003991, R01AG043434, UL1TR000448, R01EB009352. AV-45 doses were provided by Avid Radiopharmaceuticals, a wholly owned subsidiary of Eli Lilly.



