<h1 align="center"> Data Exploration with pandas — Full Topic  </h1>

### 1️⃣ Introduction to pandas
- Purpose: Learn to explore and manipulate data using pandas.
- Overview: pandas is a Python library for data manipulation and analysis, particularly for numerical tables and time series.


### 2️⃣ Installing and Importing Pandas
- pandas ইমপোর্ট করে pd হিসেবে ব্যবহার করা।

  ```python
    #!pip install pandas    # Install pandas locally
    import pandas as pd     # Import pandas
    ``` 

### 3️⃣ Importing a Dataset
- Google Drive মাউন্ট করা হয়েছে:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- CSV file (Titanic ডেটাসেট) লোড করা হয়েছে:
  ```python
  df = pd.read_csv('/content/drive/MyDrive/path_to_your_file/titanic.csv')
  ```
### 4️⃣ Exploring the Dataset
- DataFrame-এর বিভিন্ন functions দিয়ে ডেটাসেট এক্সপ্লোর করা যায়:
    ```python
    df.head()       # প্রথম 5 টি সারি দেখা।
    df.tail()       # শেষ 5 টি সারি দেখা।
    df.info()       # কলামের ডেটাটাইপ ও null values জানা।
    df.describe()   # স্ট্যাটিস্টিক্যাল সারাংশ (mean, std, min, max ইত্যাদি)।
    df.shape        # সারি ও কলামের সংখ্যা।
    df.columns      # কলাম নামসমূহ।
    df.dtypes       # প্রতিটি কলামের ডেটা টাইপ।
    ```
- **Tip**: pandas-এর display option `display.max_rows` ব্যবহার করলে, DataFrame দেখানোর সময় **সর্বাধিক কতটি রো প্রদর্শিত হবে** তা নিয়ন্ত্রণ করা যায়।  
  - Default: বড় DataFrame হলে শুধু **উপরের ও নিচের কিছু রো** দেখানো হয় এবং মাঝেরগুলো `...` দিয়ে hide করা হয়।  
  - `None` দিলে **সব রো একসাথে দেখানো হয়**।  
  - তুমি চাইলে **নির্দিষ্ট সংখ্যক রো**ও দেখতে পারো। উদাহরণস্বরূপ, উপরের ও নিচের 20 রো দেখানোর জন্য:
    ```python
    pd.set_option('display.max_rows', None) # Show all rows
    df

    pd.set_option('display.max_rows', 20)   # Show only first 20 rows
    df
    ```
  - শুধু উপরের 20 রো দেখানোর জন্য
    ```python
    df.head(20)
    ```

### 5️⃣ Accessing Specific Data
- Specific functions to access data:
  ```python
    df.nunique()        # প্রতিটি column-এ কতটি unique value আছে তা দেখাবে।
    df['Name']          # নির্দিষ্ট কলাম দেখা
    df[['Name', 'Age']] # একাধিক কলাম দেখা
    df.loc[5]           # 5 নম্বর row দেখাবে।
    df.loc[5:20]        # 5 থেকে 20 পর্যন্ত সমস্ত row দেখাবে।
    df[['Name', 'Sex']].head()  # নির্দিষ্ট কলামগুলোর প্রথম 5 row দেখাবে।
    df.loc[[1, 5, 10, 6]]       # একাধিক নির্দিষ্ট row access করা।
    dft = df.loc[5:20].copy()   # subset copy করা, যাতে মূল DataFrame পরিবর্তন না হয়।
  ```

- pandas DataFrame-এ `loc` এবং `iloc` ব্যবহার করে rows এবং columns access করা যায়।  
- **`loc`** → label-based indexing  
- **`iloc`** → integer position-based indexing 
  ```python
  df.loc[5:10, ['Name','Age']]  # Rows 5-10, specific columns
  df.iloc[0:5, 0:3]             # First 5 rows, first 3 columns
  ```

### 6️⃣ Checking & Cleaning Missing Data
- ডেটাসেটে মিসিং ভ্যালু চেক ও ক্লিন করার জন্য বিভিন্ন ফাংশন:
  ```python
    df.isnull().sum()       # প্রতিটি কলামে কতটি মিসিং ভ্যালু আছে।
    df.dropna()             # মিসিং রো বাদ দেওয়া।
    df.fillna(value)        # মিসিং জায়গায় মান বসানো।
    drop_duplicates()       # duplicate rows থাকলে সেগুলো মুছে ফেলা যায়।
  ```

### Replacing Missing Values Using Mean
- DataFrame-এ missing values পূরণের জন্য `.fillna()` ব্যবহার করা হয়।  
উদাহরণস্বরূপ, Age column-এর missing values mean দিয়ে পূরণ করা:
    ```python
    mean_value = dft['Age'].mean()      # Get the mean of Age column
    dft = dft.fillna(mean_value)        # Fill missing values with the mean
    dft.loc[dft['Age'].isnull()].head() # Check if any missing values remain in Age
    ```
### 7️⃣ Data Filtering
- শর্ত ব্যবহার:
    ```python
    df.loc[(df['Age'] < 20) & (df['Sex'] == 'female')].head()  # ফিল্টার করা মহিলা যাদের বয়স 20 এর কম
    df.loc[(df['Age'] <= 12) & (df['Survived'] == 0)].shape[0] # ফিল্টার করা 12 বছরের নিচের যাদের Survived = 0
    ```
### 8️⃣ Sorting and Grouping
- **sorting** এবং **grouping** এর জন্য:
  ```python
    df.sort_values(by='Age', ascending=True)  # বয়স অনুযায়ী সাজানো
    f.groupby('Sex')['Survived'].mean() → পুরুষ/মহিলার সার্ভাইভাল রেট তুলনা।
    ```


### 9️⃣ Value Counts and Unique Values
- categorical columns বিশ্লেষণ
  ```python
    df['Sex'].value_counts()  # প্রতিটি category কতবার আছে তা দেখায়।
    df['Pclass'].unique()     # column-এ কোন কোন unique value আছে তা list আকারে দেখায়।
  ```

### 🔟 Manipulating Columns
- df.rename() → কলাম নাম পরিবর্তন করা। 
  ```python
    df = dft.rename(columns={'Parch':'Porch'}) # Rename column 'Parch' to 'Porch'
    df.head()
  ```
- নতুন column তৈরি করা
  ```python
    df['baby'] = 1
    # Set 'baby' to 0 for rows where Age > 10
    dft.loc[dft['Age'] > 10, 'baby'] = 0
    df.head()
    ```