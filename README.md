# Alzheimers
Alzheimer’s Prediction – A Machine Learning Model Comparison
Rudy Tewelde
DSC 522 – Machine Learning II

#Introduction
Alzheimer’s Disease (AD) is a progressive disease of the brain that slowly causes impairment in memory and cognitive function. The exact cause is unknown, and no cure is available. The National Institutes of Health estimates that more than 5 million people in the United States have Alzheimer’s. Alzheimer’s is currently the sixth-leading cause of death in the US. Although young persons can and do get Alzheimer’s, the symptoms generally begin after age 60. The time from diagnosis to death can be as little as three years in persons over the age of 80.  There is a worldwide effort underway to find better ways to treat the disease, delay its onset, and prevent it from developing.

Brain Imaging via magnetic resonance imaging (MRI) is used to evaluate patients with suspected AD. Some studies have suggested that MRI features may predict the rate of decline in AD and may guide therapy in the future. However, assessing an individual’s current diagnosis can vary from one clinician to the next, or from day to day.  This study found high subjectivity and individual-level variability in cognitive assessments.

The primary goal is two-fold; one is to predict whether a patient has Alzheimer’s disease; the second is to identify individuals at risk of Alzheimer’s disease.
The aim is to develop sound models that may help clinicians catch Alzheimer’s early and predict risk factors associated with Alzheimer’s via machine learning by implementing classification algorithms for the analysis of clinical data and providing a prediction tool for the early diagnosis of the disease. The chosen methodology will be determined by fitting the data to multiple machine learning algorithms and selecting those that provide the best predictive capabilities.

#Data
The data was obtained from the Open Access Series of Imaging Studies (OASIS). OASIS is a project aimed at making MRI data sets of the brain freely available to the scientific community. OASIS is comprised of two data sets, one of longitudinal MRI Data and one of cross-sectional MRI data. These data sets are used in training multiple machine learning models to identify patients with mild to moderate dementia. Because Alzheimer’s is the most common cause of dementia, these data sets are chosen.  

Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults:

This set consists of a cross-sectional collection of 416 subjects aged 18 to 96. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. One hundred of the included subjects over the age of 60 have been clinically diagnosed with very mild to moderate Alzheimer’s disease. Additionally, a reliability data set includes 20 non-demented subjects imaged on a subsequent visit within 90 days of their initial session.

Longitudinal MRI Data in Nondemented and Demented Older Adults:

This set consists of a longitudinal collection of 150 subjects aged 60 to 96. Each subject was scanned on two or more visits, separated by at least one year for a total of 373 imaging sessions. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. Seventy-two of the subjects were characterized as non-demented throughout the study. Sixty-four of the included subjects were characterized as demented at the time of their initial visits and remained so for subsequent scans, including 51 individuals with mild to moderate Alzheimer’s disease. Another 14 subjects were characterized as non-demented at the time of their initial visit and were subsequently characterized as demented at a later visit.

The 150 patients in the longitudinal study ranged in age from 60 to 96 years. Information about their age, socioeconomic status, and education was included, along with various anatomical factors. Each patient was subject to an MRI at least twice throughout the study. Their brain volume was measured, and the MMSE (Mini-Mental State Examination) score was recorded. Each subject was then classified as either demented or non-demented.

The participants in both studies include young, middle-aged, and older adults, both demented and non-demented. Ultimately, to maintain the models’ integrity, only the longitudinal data set was utilized for analysis due to several missing observations in the cross-sectional data.




![Image](https://github.com/meronrudy/Alzheimers/blob/master/img/1.png)
