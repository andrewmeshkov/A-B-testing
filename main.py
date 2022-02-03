import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

#read data from A/B testing dataset

data = pd.read_csv('ab_data.csv')

#now we need to check the accuracy of dividing the data into groups

r = data.groupby(['group','landing_page']).count()
print(r)

#we see that there are intersections among control ans treatment , we have to fix it by clearing dataset

data = data.loc[(data['group'] == 'control') & (data['landing_page'] == 'old_page')
                | (data['group'] == 'treatment') & (data['landing_page'] == 'new_page')]

#first we will check the number of conversions in groups

groups = data.groupby(['group','landing_page','converted']).size()
groups.plot.bar()

#here we can see the ratio of groups

print(data['landing_page'].value_counts())

#for testing, we need to divide our groups into 4 parts

control = data[data['group'] == 'control']
treatment = data[data['group'] == 'treatment']

#make 4 square of separated groups

control_yes = control.converted.sum()
control_no = control.converted.size - control.converted.sum()
treatment_yes  = treatment.converted.sum()
treatment_no = treatment.converted.size - treatment.converted.sum()

prepared_groups = np.array([[control_yes, control_no], [treatment_yes, treatment_no]])

#to perform a/b testing, we will use the scipy library and the chi-square method

from scipy import stats
print(scipy.stats.chi2_contingency(prepared_groups,correction=False)[1])

control = control_yes / (control_yes + control_no)
treatment = treatment_yes / (treatment_yes + treatment_no)
print(control, treatment)

#as a result, due to the similarity of the results, we can conclude that there is no difference between the conversion rate of the two groups