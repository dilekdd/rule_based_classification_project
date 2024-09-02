import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Task_1

#Question1: Read the persona.csv file and display general information about the data set.
df = pd.read_csv("datasets/persona.csv")
df.columns
df.head()
df.info()

def quick_overview(df_summary):
    print(f"Shape Details\n{df_summary.shape}\n")
    print(f"First 5 observations\n{df_summary.head()}\n")
    print(f"Missing values by variables\n{df_summary.isnull().sum()}\n")
    print(f"Descriptive statistics for numerical variables\n{df_summary.describe()}\n")
    print(f"Overview of the DataFrame :")
    df_summary.info()

quick_overview(df)

#Question2: How many unique SOURCE are there? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

#Question3: How many unique PRICEs are there?
df["PRICE"].nunique()


#Question4: Sales amount by PRICE?
df["PRICE"].value_counts()

#Question5: Sales count by COUNTRY?
df["COUNTRY"].value_counts()

#Question6: Total price by country?
df.groupby('COUNTRY')['PRICE'].sum()

#Question7: Sales numbers by SOURCE?
df.groupby("SOURCE").size()

#Question8:Average PRICE by country?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

#Question9: Average PRICE by SOURCE type?
df.groupby("SOURCE").agg({"PRICE": "mean"})

#Question10: Average PRICE by COUNTRY and SOURCE breakdown?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})

#Task2
#Average PRICE by COUNTRY, SOURCE, SEX, AGE breakdown?
new_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
new_df.head()

#Task3
#Sort the output by PRICE
agg_df = new_df.sort_values(by="PRICE", ascending=False, ignore_index=True)
agg_df.head(5)


#Task4
#Convert the names in the index to variable names?
agg_df = new_df.sort_values(by="PRICE", ascending=False).reset_index()
agg_df.head(5)

#Task5
#Convert the Age variable to a categorical variable and add it to agg_df?
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins = [0, 18, 23, 30, 40, 70], labels = ['0_18', '19_23', '24_30', '31_40', '41_70'])
agg_df.head()

#Task6
# Define new level-based customers (personas) and add them to the data set as variables.
# The name of the new variable to be added: customers_level_based
# You need to create the customers_level_based variable by bringing together the observations in the output you will obtain in the previous question.

agg_df['customers_level_based'] = agg_df['COUNTRY'].str.upper() + '_' + agg_df['SOURCE'].str.upper() + '_' + agg_df['SEX'].str.upper() + '_' + agg_df['AGE_CAT'].astype(str)

agg_df = agg_df[['customers_level_based', 'PRICE']]

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()
agg_df.head()

#Task7
# Separate new customers (Example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Add the segments to agg_df as variables with the name SEGMENT.
# Describe the segments (Group by according to the segments and get the price mean, max, sum).

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(5)

segment_analysis = agg_df.groupby("SEGMENT", observed=False).agg({"PRICE": ["mean", "max", "sum"]})
print(segment_analysis)

#Task8
# Which segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to be earned on average?
# Which segment does a 35-year-old French woman using IOS belong to and how much income is expected to be earned on average?

new_user = "TUR_ANDROID_FEMALE_31_40"
new_user_segment = agg_df[agg_df["customers_level_based"] == new_user]

if not new_user_segment.empty:
    segment = new_user_segment['SEGMENT'].values[0]
    average_income = new_user_segment['PRICE'].values[0]
    print(
        f"33-year-old Turkish female who uses ANDROID belongs to the segment '{segment}' and the average income is {average_income:.2f}.")
else:
    print(f"No segment found for the user: {new_user}")


new_user = "FRA_IOS_FEMALE_31_40"
new_user_segment = agg_df[agg_df["customers_level_based"] == new_user]

if not new_user_segment.empty:
    segment = new_user_segment["SEGMENT"].values[0]
    average_income = new_user_segment["PRICE"].values[0]
    print(f"35 year-old French female who uses ANDROID belongs to the segment '{segment}' and the average income is {average_income:.2f}.")
else:
    print(f"No segment found for the user: {new_user}")


