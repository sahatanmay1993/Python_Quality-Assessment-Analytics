#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom
from scipy.stats import norm

df = pd.read_csv("Quality Assessment.csv")


# In[ ]:


# First 5 Rows Data
df.head()


# In[ ]:


# Number of missing values in each column
df.isnull().sum()


# In[ ]:


# Count the number of duplicate rows
df.duplicated().sum()


# In[ ]:


# Central tendency measure of the data - Median
round(df["Quantity (lts.)"].median(),3)


# In[ ]:


# Replace Null Value with Median
df["Quantity (lts.)"] = np.where(df["Quantity (lts.)"].isnull(),df["Quantity (lts.)"].median(),df["Quantity (lts.)"])


# In[ ]:


# Array of unique values
df["Assembly Line"].unique()


# In[ ]:


# Replacing Value
df["Assembly Line"].replace({"b":"B"}, inplace = True)
df["Assembly Line"].replace({"a":"A"}, inplace = True)


# In[ ]:


# Remove rows from the column has missing values (NaN)
df = df.dropna(subset=["Assembly Line"])


# In[ ]:


# Excess Time Crossed
df["Time limit Crossed"].sum()


# In[ ]:


# Difference between Maximum and Minimum
df['CO2 dissolved'].max() - df['CO2 dissolved'].min()


# In[ ]:


# Outlier in the CO2 Dissolved Data
plt.figure(figsize=(5,8))
plt.boxplot(x = df['CO2 dissolved'])
plt.title("Boxplot of CO2 Dissolved Levels")
plt.show()


# In[ ]:


# Histogram illustrating the frequency distribution of quantities
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Quantity (lts.)", kde=True,bins = 12,element='step')
plt.title('Frequency Line Chart of Quantity (lts.)')


# In[ ]:


# Outlier in the Quantity Data
plt.figure(figsize=(10, 6))
sns.boxplot(x = df['Quantity (lts.)'])
plt.title('Box Plot of Quantity (lts.)')


# In[ ]:


# Count of CO2 dissolved for each assembly line

plt.bar(df["Assembly Line"], df["CO2 dissolved"])
plt.xlabel("Assembly Line")
plt.ylabel("Count of CO2 dissolved")
plt.title("Count of CO2 dissolved for each assembly line")


# In[ ]:


# Average Value of CO0 Dissolved in Assembly Line

sns.barplot(x="Assembly Line", y="CO2 dissolved", data=df, order = ["A", "B"], errorbar = None)


# In[ ]:


# Total Amount of CO2 dissolved for each assembly line
total_CO2 = df.groupby("Assembly Line")["CO2 dissolved"].sum().reset_index()

plt.bar(x= total_CO2["Assembly Line"], height=total_CO2["CO2 dissolved"])
plt.title("CO2 dissolved for each assembly line")


# In[ ]:


# Proportion and Count of Time limit Crossed per Assembly Line
# Create a pivot table
pt = pd.pivot_table(df, values='Time limit Crossed', index='Assembly Line', aggfunc=['sum','count'], fill_value=0)
pt['pp'] = pt['sum']/pt['count']

# Select 'Assembly Line' and 'pp' columns
r = pt[['pp']].reset_index()
fig, ax1 = plt.subplots()

# Create a bar plot
plt.bar(r['Assembly Line'], r['pp'], color=['skyblue','#FFCC80']) #FFCC80 - for light orange
plt.xlabel('Assembly Line')
plt.ylabel('Proportion')

# Create a line plot for count of 'A' and 'B'
ax2 = ax1.twinx()
ax2.set_ylabel('Count')
ax2.plot(r['Assembly Line'], pt[('count')], marker='o')
ax2.tick_params(axis='y')

# Remove grid lines
ax1.grid(False)
ax2.grid(False)
plt.title('Proportion and Count of Time limit Crossed per Assembly Line')
plt.show()


# In[ ]:


# Correlation coefficient between the "CO2 dissolved" and "Quantity (lts.)"
a = df["CO2 dissolved"].corr(df["Quantity (lts.)"])
a


# In[ ]:


# Heatmap of the correlation matrix

nc = df.corr(numeric_only = True)
sns.heatmap(nc, annot = True)
plt.show()


# In[ ]:


# Computing Outlier
Q1 = df["CO2 dissolved"].quantile(0.25)
Q3 = df["CO2 dissolved"].quantile(0.75)

IQR = Q3 - Q1

lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR

print(lb)
print(ub)


# In[ ]:


# CO2 dissolved values below the lower bound (10.0635)
df[df["CO2 dissolved"] <10.0635]


# In[ ]:


# CO2 dissolved values below the upper bound (18.18534705)
df[df["CO2 dissolved"] >18.18534]


# In[ ]:


# Values without Outlier or within Bound
filtered_df = df[(df["CO2 dissolved"] >= 10.0635) & (df["CO2 dissolved"] <= 18.18534)]
filtered_df


# In[ ]:


# Average of CO0 Dissolved
filtered_df["CO2 dissolved"].mean()


# # Probability

# In[ ]:


# Calculate the frequency of "Time limit crossed" for each assembly line
time_limit_counts = df.groupby("Assembly Line")["Time limit Crossed"].value_counts()

# Calculate the total number of occurrences for each assembly line
assembly_line_counts = df["Assembly Line"].value_counts()

# Calculate the probability of "Time limit crossed" for each assembly line
probabilities = time_limit_counts / assembly_line_counts

# Display the probabilities
print("Probabilities of Time limit crossed per assembly line:")
print(probabilities)


# OR


# In[ ]:


grouped = df.groupby('Assembly Line')['Time limit Crossed'].value_counts(normalize=True).reset_index(name='Probability')
grouped


# ## Now considering the probabilities calculated in previous questions, calculate the probability that 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line B . Round it off to 2 decimal places.

# In[ ]:


from scipy.stats import binom

k = 10
n = 50
p = 0.2125

prob_b = binom.pmf(k,n,p)
prob_b


# ## Calculate the probability that 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line A. Round it off to 2 decimal places.

# In[ ]:


from scipy.stats import binom

k = 10
n = 50
p = 0.143885

prob_a = binom.pmf(k,n,p)
prob_a


# ## Calculate the probability that at least 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line B . Round it off to 2 decimal places.

# In[ ]:


from scipy.stats import binom

p = 0.212500
n = 50
a = 0

for i in range(10):
    pmf = binom.pmf(i,n,p)
    print(pmf)
    a+=pmf
print("The probability that at least 10 out of a sample of 50 bottle crossed the time limit",round(1-a,2))


# In[ ]:


from scipy.special import binom

# Probabilities of exceeding the time limit for Assembly line B
p_success = 0.2125  # Probability of a bottle exceeding the limit (case 1)
p_failure = 1 - p_success  # Probability of a bottle not exceeding the limit (case 0)

# Desired number of successes (at least 10) and total number of trials (50 bottles)
k = 10  # Minimum number of successes (at least 10)
n = 50  # Total number of trials (50 bottles)

# Calculate the probability using the binomial probability formula
probability = 0

# Loop to calculate the probability for k successes (10 to 50)
for i in range(k, n + 1):
    probability += binom(n, i) * (p_success**i) * (p_failure**(n-i))

# Round the probability to 2 decimal places
probability = round(probability, 2)

print(f"The probability of at least 10 bottles crossing the time limit is: {probability}")


# In[ ]:


from scipy.stats import binom

# Given probabilities
p_success = 0.212500  # Probability of a bottle crossing the time limit on Assembly line B
n = 50  # Total number of bottles in the sample

# Calculate the probability of having less than 10 successes
p_less_than_10 = binom.cdf(9, n, p_success)

# Calculate the probability of having at least 10 successes
p_at_least_10 = 1 - p_less_than_10

# Round the result to 2 decimal places
p_at_least_10 = round(p_at_least_10, 2)

print("Probability of at least 10 bottles crossing the time limit on Assembly line B:", p_at_least_10)


# In[ ]:


from scipy.stats import binom

p = 0.212500
n=50
k=9

cdf = 1 - binom.cdf(k, n, p)
print(round(cdf,2))


# ## Please calculate the probability that at least 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line A . Round it off to 2 decimal places.

# In[ ]:


from scipy.stats import binom

p = 0.143885
n = 50
a = 0

for i in range(10):
    pmf = binom.pmf(i,n,p)
    print(pmf)
    a+=pmf
print("The probability that at least 10 out of a sample of 50 bottle crossed the time limit",round(1-a,2))


# In[ ]:


from scipy.special import binom

# Probabilities of exceeding the time limit for Assembly line B
p_success = 0.143885  # Probability of a bottle exceeding the limit (case 1)
p_failure = 1 - p_success  # Probability of a bottle not exceeding the limit (case 0)

# Desired number of successes (at least 10) and total number of trials (50 bottles)
k = 10  # Minimum number of successes (at least 10)
n = 50  # Total number of trials (50 bottles)

# Calculate the probability using the binomial probability formula
probability = 0

# Loop to calculate the probability for k successes (10 to 50)
for i in range(k, n + 1):
    probability += binom(n, i) * (p_success**i) * (p_failure**(n-i))

# Round the probability to 2 decimal places
probability = round(probability, 2)

print(f"The probability of at least 10 bottles crossing the time limit is: {probability}")


# In[ ]:


from scipy.stats import binom

# Given probabilities
p_success = 0.143885  # Probability of a bottle crossing the time limit on Assembly line B
n = 50  # Total number of bottles in the sample

# Calculate the probability of having less than 10 successes
p_less_than_10 = binom.cdf(9, n, p_success)

# Calculate the probability of having at least 10 successes
p_at_least_10 = 1 - p_less_than_10

# Round the result to 2 decimal places
p_at_least_10 = round(p_at_least_10, 2)

print("Probability of at least 10 bottles crossing the time limit on Assembly line B:", p_at_least_10)


# # Estimation

# # In 2 Litre soft drink bottles, the drink filled is close to normally distributed. If bottles contain less than 95% of the listed net content (around 1.90 litres), the manufacturer may be penalised by the state office of consumer affairs. Bottles that have a net quantity above 2.1 litres may cause excess spillage upon opening

# ## What is the probability that the bottle content can be either penalised or have spillage? Round it off to 2 decimal places. [Hint: Use mean and SD for quantity variable after outlier removal]

# In[ ]:


Q1 = df["Quantity (lts.)"].quantile(0.25)
Q3 = df["Quantity (lts.)"].quantile(0.75)

IQR = Q3 -Q1

lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR

print(lb)
print(ub)


# In[ ]:


df[(df["Quantity (lts.)"] < lb) & (df["Quantity (lts.)"] > ub)]


# In[ ]:


print(df["Quantity (lts.)"].mean())
print(np.std(df["Quantity (lts.)"]))


# In[ ]:


import pandas as pd
from scipy import stats

mean = 1.9979
std_dev = 0.0480

# Calculating z-scores
z_score_penalty = (1.9 - mean) / std_dev
z_score_spillage = (2.1 - mean) / std_dev

# Calculating probabilities
prob_penalty = stats.norm.cdf(z_score_penalty) # Calculate the probability of content less than 1.90 liters
prob_spillage = 1 - stats.norm.cdf(z_score_spillage) # Calculate the probability of content greater than 2.1 liters

# Total probability
total_prob = round((prob_penalty + prob_spillage), 2)
total_prob


# In[ ]:


from scipy.stats import norm

# Given mean and SD after outlier removal
mean = 1.9979  # Mean quantity in liters
sd = 0.0480  # Standard deviation in liters

# Calculate the probability of content less than 1.90 liters
p_less_than_1_90 = norm.cdf(1.90, mean, sd)

# Calculate the probability of content greater than 2.1 liters
p_greater_than_2_10 = 1 - norm.cdf(2.10, mean, sd)

# Calculate the total probability of either penalization or spillage
p_total = p_less_than_1_90 + p_greater_than_2_10

# Round the result to 2 decimal places
p_total = round(p_total, 2)

print("Probability of bottle content being either penalized or causing spillage:", p_total)


# In[ ]:


from scipy.stats import norm

# Penalty threshold (as a percentage of listed content)
penalty_threshold = 0.95

# Spillage threshold
spillage_threshold = 2.1

# Mean and standard deviation (replace with your actual values)
mean = 1.9979  # Litres (example value)
std_dev = 0.0480  # Litres (example value)

# Calculate standardized thresholds (z-scores)
penalty_z = (penalty_threshold * 2.0 - mean) / std_dev
spillage_z = (spillage_threshold - mean) / std_dev

# Calculate probability of falling below penalty threshold (penalty tail)
p_penalty = norm.cdf(penalty_z, loc=0, scale=1)  # CDF with mean 0 and std dev 1

# Calculate probability of exceeding spillage threshold (spillage tail)
p_spillage = 1 - norm.cdf(spillage_z, loc=0, scale=1)  # 1 - CDF for upper tail

# Total probability (penalty or spillage)
total_prob = p_penalty + p_spillage

# Round the probability to 2 decimal places
total_prob = round(total_prob, 2)

print(f"The probability of a bottle being penalized or causing spillage is: {total_prob}")


# ## What is the probability that the bottle content is in between 1.95 litres and 2.05 litres? Round it off to 2 decimal places.

# In[ ]:


from scipy.stats import norm

# Given mean and SD after outlier removal
mean = 1.9979  # Mean quantity in liters
sd = 0.0480  # Standard deviation in liters

# Calculate the probability of content less than 2.05 liters
p_less_than_2_05 = norm.cdf(2.05, mean, sd)

# Calculate the probability of content less than 1.95 liters
p_less_than_1_95 = norm.cdf(1.95, mean, sd)

# Calculate the probability of content being in between 1.95 liters and 2.05 liters
p_between_1_95_and_2_05 = p_less_than_2_05 - p_less_than_1_95

# Round the result to 2 decimal places
p_between_1_95_and_2_05 = round(p_between_1_95_and_2_05, 2)

print("Probability of bottle content being in between 1.95 liters and 2.05 liters:", p_between_1_95_and_2_05)


# In[ ]:


import pandas as pd
from scipy import stats

mean = 1.9979
std_dev = 0.0480

# Calculating z-scores
z_score_penalty = (1.95 - mean) / std_dev
z_score_spillage = (2.05 - mean) / std_dev

# Calculating probabilities
prob_penalty = stats.norm.cdf(z_score_penalty) # Calculate the probability of content less than 1.90 liters
prob_spillage = stats.norm.cdf(z_score_spillage) # Calculate the probability of content less than 2.1 liters

# Total probability
total_prob = round((prob_spillage - prob_penalty), 2)
total_prob


# ## Calculate the 90% interval estimate for the Quantity variable.

# In[ ]:


from scipy.stats import norm
import numpy as np

confidence_level = 0.90
sample_size = len(df)

standard_error = np.std(df["Quantity (lts.)"])
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df = sample_size - 1 ) * standard_error

interval_e = (np.mean(df["Quantity (lts.)"]) - margin_of_error, np.mean(df["Quantity (lts.)"]) + margin_of_error)
interval_e


# In[ ]:




