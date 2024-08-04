                                             # Data analysis Using Python Tasks (1 Month tasks)
                                                  # Create 1D, 2D, and 3D arrays:
import numpy as np

# 1D array
array_1d = np.array([1, 2, 3, 4, 5])
# 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
# 3D array
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("1D Array:", array_1d)
print("2D Array:", array_2d)
print("3D Array:", array_3d)


                                                # Calculate mean, median, mode, and standard deviation:

from scipy import stats

# Mean
mean_1d = np.mean(array_1d)
# Median
median_1d = np.median(array_1d)
# Mode
mode_1d = stats.mode(array_1d)
# Standard deviation
std_1d = np.std(array_1d)

print("Mean:", mean_1d)
print("Median:", median_1d)
print("Mode:", mode_1d)
print("Standard Deviation:", std_1d)


                                                        #Task 2 (1st Week): Pandas DataFrame
                                                        #Create a dictionary and DataFrame:

import pandas as pd
import numpy as np

data = {
    'ord_no': [70001, np.nan, 70002, 70004, np.nan, 70005, np.nan, 70010, 70003, 70012, np.nan, 70013],
    'purch_amt': [150.5, 270.65, 65.26, 110.5, 948.5, 2400.6, 5760, 1983.43, 2480.4, 250.45, 75.29, 3045.6],
    'ord_date': ['2012-10-05', '2012-09-10', np.nan, '2012-08-17', '2012-09-10', '2012-07-27', '2012-09-10', 
                 '2012-10-10', '2012-10-10', '2012-06-27', '2012-08-17', '2012-04-25'],
    'customer_id': [3002, 3001, 3001, 3003, 3002, 3001, 3001, 3004, 3003, 3002, 3001, 3001],
    'salesman_id': [5002, 5003, 5001, np.nan, 5002, 5001, 5001, np.nan, 5003, 5002, 5003, np.nan]
}

df = pd.DataFrame(data)


                         #Handle missing values:        
                                                        
# Print Missing values
print("Missing values in DataFrame:")
print(df.isnull().sum())

# Convert to CSV and read back
df.to_csv('csv1.csv', index=False)
df_from_csv = pd.read_csv('csv1.csv')

# Identify columns with at least one missing value
print("Columns with at least one missing value:")
print(df.columns[df.isnull().any()])

# Count the number of missing values in each column
print("Number of missing values in each column:")
print(df.isnull().sum())

# Replace missing values with NaN (they are already NaN)
df.fillna(np.nan, inplace=True)

# Drop rows/columns with missing values
df_dropped_rows = df.dropna()
df_dropped_cols = df.dropna(axis=1)

# Drop rows where all elements are missing
df_dropped_all = df.dropna(how='all')

# Keep rows with at least 2 NaN values
df_at_least_2_nan = df[df.isnull().sum(axis=1) >= 2]

# Total missing values in DataFrame
total_missing = df.isnull().sum().sum()
print("Total missing values in DataFrame:", total_missing)


                                                            #Task 3 (1st Week): Matplotlib Charts
                                          
import matplotlib.pyplot as plt

languages = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]

plt.bar(languages, popularity)
plt.xlabel('Programming languages')
plt.ylabel('Popularity')
plt.title('Popularity of Programming Languages')
plt.show()

               #Pie chart:
plt.pie(popularity, labels=languages, autopct='%1.1f%%')
plt.title('Popularity of Programming Languages')
plt.show()
