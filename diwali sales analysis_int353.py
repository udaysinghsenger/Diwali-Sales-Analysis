#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION

# Diwali, also known as the Festival of Lights, is one of the most celebrated and
# widely observed festivals in India and other parts of the world. Beyond its
# cultural and religious significance, Diwali also marks a significant period for
# businesses, with a surge in consumer spending during the festive season. The
# Diwali Sales Analysis Exploratory Data Analysis (EDA) Project aims to shed
# light on the trends, patterns, and insights within the sales data during this festive
# period.
# 
# The primary focus of this Diwali Sales Analysis EDA Project is to unravel
# invaluable insights from the dataset, focusing on key aspects such as
# geographical regions where sales are at their peak. By scrutinizing
# the data, we aim to find not only the geographic hotspots but also the
# specific products that garner heightened consumer interest. Additionally, a
# nuanced exploration into the gender-based purchasing patterns will be
# conducted to identify which demographic contributes significantly to the festive
# sales.

# # DOMAIN KNOWLEDGE

# Diwali Sales Analysis is a field that studies how businesses can understand and improve their sales during the Diwali festival.
# Diwali is a major celebration in India and other parts of the world, and it is a time when people spend more money on gifts and other items.
# 
# Businesses can use data analysis to learn about consumer behavior during Diwali, and then use this information to plan their marketing and sales strategies.
# Diwali Sales
# Analysis is a complex field that requires knowledge of business, marketing, and cultural factors.

#  # WHY I CHOOSE THIS DATASET
# 

# 1. Cultural and Economic Significance: 
# Diwali is a major cultural and religious festival in various regions, especially in India. Analyzing Diwali sales data provides insights into consumer behavior during a period of heightened spending and festivities.
# 
# 
# 2. Geographical Insights: 
# The dataset can offer valuable information about regional variations in sales during Diwali. 
# 
# 
# 3. Target Customer Identification:
# Analyzing gender-based and demographic preferences during Diwali can aid in identifying target customer segments. This
# knowledge is crucial for businesses 
# 
# 
# 4. Strategic Decision-Making: 
# The insights derived from the dataset can guide businesses in making informed strategic decisions for future Diwali seasons.
# 
# 

# # IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# 1. Pandas: 
# Used for handling and organizing structured data, Pandas facilitates
# tasks such as loading datasets, data cleaning, and basic exploratory data
# analysis.
# 2. NumPy: 
# Essential for numerical operations and computations, NumPy is
# employed to work with numerical data efficiently, providing support for arrays
# and mathematical functions.
# 3. Matplotlib: 
# This library is instrumental for creating a diverse range of static
# visualizations, enabling the representation of trends and patterns in the Diwali
# sales dataset.
# 4. Seaborn: 
# Built on top of Matplotlib, Seaborn offers a high-level interface for
# generating aesthetically pleasing statistical graphics. It is employed to enhance
# the visual interpretation of the data.
# 5.SciPy: 
# This library was used for hypothesis testing.

# In[2]:


df = pd.read_csv('Diwali Sales Data1.csv', encoding= 'unicode_escape')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe(include=object)


# # Dropping the null columns

# In[9]:


df.drop(["Status","unnamed1"], axis=1, inplace=True)


# In[10]:


df.isnull().sum()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.isnull().sum()


# # OUTLIERS

# In[13]:


df.boxplot()
plt.xticks(rotation=90)
plt.figure(figsize=(20,15))


# In[14]:


sns.boxplot(df['Age'])
plt.figure(figsize=(8,6))


# In[15]:


q1=df['Age'].quantile(0.25)
q3=df['Age'].quantile(0.75)
iqr=q3-q1
low=q1-1.5*iqr
upp=q3+1.5*iqr


# In[16]:


Outliers = []
df['Age']
for i in df['Age']:
    if ((i > upp) or (i < low)):
        Outliers.append(i)


# In[17]:


filtered_data = df[(df['Age'] >= low) & (df['Age'] <= upp)]
sns.boxplot(x=filtered_data['Age'])


# # Changing the data type

# In[18]:


df['Amount']=df['Amount'].astype('int')


# In[19]:


df['Amount'].dtypes


# In[20]:


df.columns


# In[21]:


df=df.rename(columns={'Age Group':'Age_Group'})


# In[22]:


df.head()


# # DATA VISUALIZATION

# ## Q1.WHAT IS THE NUMBER  OF PURCHASE DONE BY DIFFERENT GENDER

# In[23]:


ax=sns.countplot(x='Gender',data=df)

for bars in ax.containers:
    ax.bar_label(bars)


# ## Q2. WHAT IS THE AMOUNT SPEND BY DIFFERNT GENDERS ON PURCHACHING?

# In[24]:


df.groupby(['Gender'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)


# In[25]:


sales_gen=df.groupby(['Gender'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)

sns.barplot(x='Gender',y='Amount',data=sales_gen)


# ### From above graphs we can see that most of the buyers are females and even the purchasing power of females are greter than men

# ## Q3.WHICH AGE GROUP HAS HEIGHEST NUMBER OF PURCHASES?

# In[26]:


sns.countplot(data=df,x='Age_Group')


# ## Q4. How does the distribution of 'Age Group' vary between different genders??

# In[27]:


ax=sns.countplot(data=df,x='Age_Group',hue='Gender')

for bars in ax.containers:
    ax.bar_label(bars)


# ## Q5. WHAT IS THE AMOUNT SPEND BY DIFFERNT AGE GROUP ON PURCHACHING?

# In[28]:


sales_age=df.groupby(['Age_Group'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)
sns.barplot(x='Age_Group',y='Amount',data=sales_age)


# ##### From above graph we can see that most of the buyers are of the age group betwwen 26-35 years female

# # STATE

# ## Q6 What is the distribution of customers across different states?

# In[51]:


sns.countplot(x='State',data=df)
sns.set(rc={'figure.figsize':(36,15)})


# ## Q7 Which states have the highest number of orders?

# In[30]:


sales_state=df.groupby(['State'],as_index=False)['Orders'].sum().sort_values(by='Orders',ascending=False).head(10)
sns.set(rc={'figure.figsize':(15,5)})

sns.barplot(x='State',y='Orders',data=sales_state)


# ## Q8 WHAT IS THE AMOUNT SPEND BY DIFFERNT STATES ON PURCHACHING?

# In[31]:


sales_amount=df.groupby(['State'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)
sns.set(rc={'figure.figsize':(15,5)})

sns.barplot(x='State',y='Amount',data=sales_amount)


# # Zone

# ## Q9 What is the distribution of customers across different zones?

# In[32]:


plt.figure(figsize=(8,8))
df['Zone'].value_counts().plot.pie(autopct='%1.2f%%')
plt.show()


# ## Q10. WHAT IS THE AMOUNT SPEND BY PEOPLE OF DIFFERNT ZONES ON PURCHACHING?

# In[33]:


zone_sales = df.groupby('Zone')['Amount'].sum()
plt.figure(figsize=(10, 6))
zone_sales.plot(kind='bar', color='skyblue')
plt.title('Zone-wise Sales Comparison')
plt.xlabel('Zone')
plt.ylabel('Total Sales Amount')
plt.show()


# ## MARITAL STATUS

# ## Q11 What is the distribution of customers' marital status in the dataset

# In[49]:


sns.countplot(x='Marital_Status',data=df)
sns.set(rc={'figure.figsize':(5,5)})


# ## Q12 Which combinations of marital status and gender contribute the most to the total sales amounts

# In[35]:


sales_ma=df.groupby(['Marital_Status','Gender'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)
sns.set(rc={'figure.figsize':(7,5)})

sns.barplot(data=sales_ma, x='Marital_Status', y='Amount',hue='Gender')


# ###### From the graph we can see that most of the buyers are married (women)and thet have high purchase power

# # Occupation 

# ## Q13.WHAT IS THE NUMBER OF PURCHASE MADE BY DIFFERENT OCCCUPATION

# In[36]:


sns.set(rc={'figure.figsize':(20,10)})
ax=sns.countplot(x='Occupation',data=df)

for bars in ax.containers:
    ax.bar_label(bars)


#  ## Q14. WHAT IS THE AMOUNT SPEND BY DIFFERNT OCCUPATIONS ON PURCHACHING?

# In[37]:


sales_occ=df.groupby(['Occupation'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False)
sns.set(rc={'figure.figsize':(20,9)})

sns.barplot(data=sales_occ, x='Occupation', y='Amount')


# ##### From the we cn see the most of the buyer are from IT,Healthcare,aviation sector

# # Product Category

# ## Q15 What is the distribution of products across different product categories?

# In[38]:


sns.set(rc={'figure.figsize':(25,9)})
ax=sns.countplot(x='Product_Category',data=df)

for bars in ax.containers:
    ax.bar_label(bars)


# ## Q16 Which product categories contribute the most to the total sales amounts

# In[39]:


sales_pro=df.groupby(['Product_Category'],as_index=False)['Amount'].sum().sort_values(by='Amount',ascending=False).head(10)
sns.set(rc={'figure.figsize':(20,9)})

sns.barplot(data=sales_pro, x='Product_Category', y='Amount')


# ##### above graph we can see that most of the sold product are from Food ,Clothing  and Electronics category

# ## Q17.What is the distribution of average spending across various product categories in different zones

# In[40]:


plt.figure(figsize=(23, 6))
sns.barplot(x="Product_Category", y="Amount", hue="Zone", data=df)
plt.title("Average Amount Spent by Product Category in Each Zone")
plt.xlabel("Product Category")
plt.ylabel("Average Amount")
plt.xticks(rotation=45)
plt.legend(title="Zone")
plt.show()


# ## Q18. What correlations can be observed among the numerical columns 'Age,' 'Orders,' and 'Amount' in the dataset

# In[41]:


numerical_columns = ['Age', 'Orders', 'Amount']

correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ## Q19 What is the overall correlation among the numerical columns in the dataset?

# In[42]:


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap",fontsize=25)
plt.show()


# ## Q20.Which 'Product_ID' is the most popular or has the highest number of purchases in the dataset?"

# In[43]:


most_popular_product = df['Product_ID'].value_counts().idxmax()
print("The most popular Product_ID is:" ,most_popular_product)


# # Correlation hypothesis testing 

# In[44]:


age = df['Age']
amount = df['Amount']

# Calculate the Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(age, amount)

# Set the significance level (alpha)
alpha = 0.05

print(f"Pearson Correlation Coefficient: {correlation_coefficient:.4f}")
print(f"P-Value: {p_value:.4f}")

if p_value < alpha:
    print("Reject the null hypothesis (Ha): There is a significant correlation between Age and Amount.")
else:
    print("Fail to reject the null hypothesis (H0): There is no significant correlation between Age and Amount.")


# In[45]:


sns.pairplot(df[['Age', 'Amount']])
plt.show()


# # LIMITATIONS

# 1. Limited Variables: The analysis was constrained by the available variables in
# the dataset. Not having access to additional variables limited the scope of the
# analysis.
# 
# 2. Missing Values :The dataset had a few missing values. While we handled these by removing the
#  rows containing null values , this approach may introduce some bias into the data.
# 
# 3. Outliers :The presence of outliers can skew the results of the analysis. While we used the IQR
#  method to detect and handle outliers, this method may not capture all outliers
#  

# # RECOMMENDATIONS

# 1.Handling Missing Values :
# Explore other methods for handling missing values, such as interpolation or multiple imputations, which may provide more accurate results.
#  
# 2.Outlier Detection :
# Consider using more robust outlier detection methods, such as the Z-score method or the Modified Z-score method.
#  
# 3.Model Selection and Optimization :
# Experiment with different machine learning algorithms and optimization techniques to improve the performance of the predictive model
# 

# # CONCLUSION

# ### Married women in  age group  of 26-35 yrs from "UP" , "Maharastra" and "karnataka" working in IT,Healthcare and Aviation are more likely to buy products from Food,Clothing and Electronics Category

# In[ ]:




