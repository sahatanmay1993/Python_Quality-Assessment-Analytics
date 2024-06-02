import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom
from scipy.stats import norm

df = pd.read_csv("Quality Assessment.csv")

# First 5 Rows Data

df.head()

# Number of missing values in each column

df.isnull().sum()


# Count the number of duplicate rows

df.duplicated().sum()


# Central tendency measure of the data - Median

round(df["Quantity (lts.)"].median(),3)


# Replace Null Value with Median

df["Quantity (lts.)"] = np.where(df["Quantity (lts.)"].isnull(),df["Quantity (lts.)"].median(),df["Quantity (lts.)"])


# Array of unique values

df["Assembly Line"].unique()


# Replacing Value

df["Assembly Line"].replace({"b":"B"}, inplace = True)
df["Assembly Line"].replace({"a":"A"}, inplace = True)


# Remove rows from the column has missing values (NaN)

df = df.dropna(subset=["Assembly Line"])


# Excess Time Crossed

df["Time limit Crossed"].sum()


# Difference between Maximum and Minimum

df['CO2 dissolved'].max() - df['CO2 dissolved'].min()


# Outlier in the CO2 Dissolved Data

plt.figure(figsize=(5,8))
plt.boxplot(x = df['CO2 dissolved'])
plt.title("Boxplot of CO2 Dissolved Levels")
plt.show()


# Histogram illustrating the frequency distribution of quantities

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Quantity (lts.)", kde=True,bins = 12,element='step')
plt.title('Frequency Line Chart of Quantity (lts.)')


# Outlier in the Quantity Data

plt.figure(figsize=(10, 6))
sns.boxplot(x = df['Quantity (lts.)'])
plt.title('Box Plot of Quantity (lts.)')


# Count of CO2 dissolved for each assembly line

plt.bar(df["Assembly Line"], df["CO2 dissolved"])
plt.xlabel("Assembly Line")
plt.ylabel("Count of CO2 dissolved")
plt.title("Count of CO2 dissolved for each assembly line")


# Average Value of CO2 Dissolved in Assembly Line

sns.barplot(x="Assembly Line", y="CO2 dissolved", data=df, order = ["A", "B"], errorbar = None)


# Total Amount of CO2 dissolved for each assembly line

total_CO2 = df.groupby("Assembly Line")["CO2 dissolved"].sum().reset_index()

plt.bar(x= total_CO2["Assembly Line"], height=total_CO2["CO2 dissolved"])
plt.title("CO2 dissolved for each assembly line")


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


# Correlation coefficient between the "CO2 dissolved" and "Quantity (lts.)"

a = df["CO2 dissolved"].corr(df["Quantity (lts.)"])
a


# Heatmap of the correlation matrix

nc = df.corr(numeric_only = True)
sns.heatmap(nc, annot = True)
plt.show()


# Computing Outlier

Q1 = df["CO2 dissolved"].quantile(0.25)
Q3 = df["CO2 dissolved"].quantile(0.75)

IQR = Q3 - Q1

lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR

print(lb)
print(ub)


# CO2 dissolved values below the lower bound (10.0635)

df[df["CO2 dissolved"] <10.0635]


# CO2 dissolved values below the upper bound (18.18534705)

df[df["CO2 dissolved"] >18.18534]


# Values without Outlier or within Bound

filtered_df = df[(df["CO2 dissolved"] >= 10.0635) & (df["CO2 dissolved"] <= 18.18534)]
filtered_df


# Average of CO0 Dissolved

filtered_df["CO2 dissolved"].mean()


# # Probability

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

grouped = df.groupby('Assembly Line')['Time limit Crossed'].value_counts(normalize=True).reset_index(name='Probability')
grouped


# ## Now considering the probabilities calculated in previous questions, calculate the probability that 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line B . Round it off to 2 decimal places.

from scipy.stats import binom

k = 10
n = 50
p = 0.2125

prob_b = binom.pmf(k,n,p)
prob_b


# ## Calculate the probability that 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line A. Round it off to 2 decimal places.

from scipy.stats import binom

k = 10
n = 50
p = 0.143885

prob_a = binom.pmf(k,n,p)
prob_a


# ## Calculate the probability that at least 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line B . Round it off to 2 decimal places.

from scipy.stats import binom

p = 0.212500
n = 50
a = 0

for i in range(10):
    pmf = binom.pmf(i,n,p)
    print(pmf)
    a+=pmf
print("The probability that at least 10 out of a sample of 50 bottle crossed the time limit",round(1-a,2))

# OR

from scipy.stats import binom

p = 0.212500
n=50
k=9

cdf = 1 - binom.cdf(k, n, p)
print(round(cdf,2))


# ## Calculate the probability that at least 10 bottles crossed the time limit out of a sample of 50 bottles on Assembly line A . Round it off to 2 decimal places.

from scipy.stats import binom

p = 0.143885
n = 50
a = 0

for i in range(10):
    pmf = binom.pmf(i,n,p)
    print(pmf)
    a+=pmf
print("The probability that at least 10 out of a sample of 50 bottle crossed the time limit",round(1-a,2))


# # Estimation

# # In 2 Litre soft drink bottles, the drink filled is close to normally distributed. If bottles contain less than 95% of the listed net content (around 1.90 litres), the manufacturer may be penalised by the state office of consumer affairs. Bottles that have a net quantity above 2.1 litres may cause excess spillage upon opening

# ## What is the probability that the bottle content can be either penalised or have spillage? Round it off to 2 decimal places.

Q1 = df["Quantity (lts.)"].quantile(0.25)
Q3 = df["Quantity (lts.)"].quantile(0.75)

IQR = Q3 -Q1

lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR

print(lb)
print(ub)

# Ouantity filtering Outlier

df[(df["Quantity (lts.)"] < lb) & (df["Quantity (lts.)"] > ub)]

# Average and Standard Deviation of Quantity

print(df["Quantity (lts.)"].mean())
print(np.std(df["Quantity (lts.)"]))


# In 2 Litre soft drink bottles, the drink filled is close to normally distributed. 
# If bottles contain less than 95% of the listed net content (around 1.90 litres), the manufacturer may be penalised by the state office of consumer affairs. 
# Bottles that have a net quantity above 2.1 litres may cause excess spillage upon opening.

# What is the probability that the bottle content can be either penalised or have spillage? Round it off to 2 decimal places. 

import pandas as pd
from scipy import stats

mean = 1.9979
std_dev = 0.0480

# Calculating z-scores
z_score_penalty = (1.9 - mean) / std_dev
z_score_spillage = (2.1 - mean) / std_dev

# Calculating probabilities
prob_penalty = stats.norm.cdf(z_score_penalty)
prob_spillage = 1 - stats.norm.cdf(z_score_spillage)

# Total probability
total_prob = round((prob_penalty + prob_spillage), 2)
total_prob


# ## What is the probability that the bottle content is in between 1.95 litres and 2.05 litres? Round it off to 2 decimal places.

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

from scipy.stats import norm
import numpy as np

confidence_level = 0.90
sample_size = len(df)

standard_error = np.std(df["Quantity (lts.)"])
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df = sample_size - 1 ) * standard_error

interval_e = (np.mean(df["Quantity (lts.)"]) - margin_of_error, np.mean(df["Quantity (lts.)"]) + margin_of_error)
interval_e


