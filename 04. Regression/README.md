<h1 align="center">Regression in Machine Learning</h1>

### 1️⃣ Goal 
- টার্গেট ভেরিয়েবল বা আউটপুটের continuous মান (সংখ্যাগত ফলাফল) predict করা।
- ইনপুট ফিচারগুলোর সাথে আউটপুটের সম্পর্ক শেখা এবং একটি mathematical model তৈরি করা।
- নতুন বা unseen ডেটার উপর accurate prediction  
- বাস্তব জীবনের উদাহরণ:
    - বাড়ির মূল্য অনুমান করা (House Price Prediction)
    - গাড়ির mileage বা fuel efficiency অনুমান করা
    - বিক্রয় (Sales) বা স্টক প্রাইস পূর্বানুমান করা
- Regression মডেল দিয়ে আমরা decision making ও forecasting সহজভাবে করতে পারি।
  
### 2️⃣ Importing Necessary Libraries
- লাইব্রেরি :
  ```python
    import numpy as np
    import pandas as pd
  ```

### 3️⃣ Importing a Dataset
- Google Drive মাউন্ট করা হয়েছে:
  ```python
    from google.colab import drive
    drive.mount('/content/drive') # এটি ব্যবহার করা হয় যাতে গুগল ড্রাইভ থেকে ফাইল (ডেটাসেট) সরাসরি লোড করা যায়।
  ```
- dataset load করার জন্য pandas এর read_csv() ফাংশন ব্যবহার করা।
  ```python
   df = pd.read_csv('/content/drive/MyDrive/path_to_your_file/garments_worker_productivity.csv')
   df
  ```
### 4️⃣ Exploring the Dataset
- day কলামের সকল ইউনিক মানগুলো দেখতে পারবে।
  ```python
     df['day'].unique()
  ```

### 5️⃣ Target Variable এবং Features
- actual_productivity হল আমাদের যেটা predict করতে হবে সেই আউটপুট ভেরিয়েবল।
- এখানে X হলো feature matrix, যেটা মডেল শেখার জন্য ব্যবহার হবে। (সমস্ত সংখ্যাগত এবং categorical ফিচার একত্রে রাখা হয়েছে)
  ```python
    y = df['actual_productivity']           # Target variable
    # Feature Selection
    X = df[['quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']]
    X
  ```
### 6️⃣ Train-Test Split
- Dataset কে 70% train এবং 30% test set এ ভাগ করা।
- .isnull().sum() প্রতিটি কলামে কতটা missing value আছে তা দেখায়।
  ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
    )

    X_train.isnull().sum()
  ```

### 7️⃣ Handling Missing Values
- **SimpleImputer** ব্যবহার করে আমরা wip কলামের missing values পূরণ করেছি।
- strategy='mean' মানে কলামের গড় মান দিয়ে missing value পূরণ করা।
- fit_transform() দিয়ে আমরা training set-এ apply করেছি,
- transform() দিয়ে testing set-এ apply করেছি।
  ```python
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')

    X_train['wip'] = imputer.fit_transform(X_train[['wip']])
    X_test['wip'] = imputer.transform(X_test[['wip']])

    X_train.isnull().sum()      # checking missing values after imputation
  ```
### 8️⃣ Encoding Categorical Columns
- LabelEncoder ব্যবহার করে **categorical variables** কে **numerical values** এ রূপান্তর করা হয়েছে।
- Column-wise fit_transform training set-এ ব্যবহার করা হয়েছে এবং transform testing set-এ।
- এখন categorical features মেশিন লার্নিং মডেলে ব্যবহারযোগ্য।
  ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    # Label Encode only the categorical columns using fit_transform and transform
    categorical_columns = ['quarter', 'day', 'department']

    for column in categorical_columns:
        X_train[column] = le.fit_transform(X_train[column])
        X_test[column] = le.transform(X_test[column])

    X_train          # checking encoded training features 
  ```

### 9️⃣ Scaling Numerical Features
- **StandardScaler** ব্যবহার করে numerical columns কে **standardize** করা হয়েছে।
- Scaling করা হলে model **faster convergence** এবং **better performance** পায়।
  ```python
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()

    # Normalize only the numerical columns using fit_transform and transform
    numerical_columns = ['targeted_productivity', 'smv', 'wip', 'over_time', 
                     'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers']

    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    X_train      # checking scaled training features
  ```

### ❗Handling Missing Values, Encoding Categorical Columns, Scaling Numerical Features - এই তিনটি ধাপের পর ডেটা এখন পুরোপুরি preprocessed এবং ready, Regression model train করার জন্য।


### 1️⃣0️⃣ Training a Regression Model
- LinearRegression() – সরল লিনিয়ার রিগ্রেশন।
DecisionTreeRegressor() – non-linear relationships ধরতে পারে।
- RandomForestRegressor() – multiple decision trees মিলিয়ে ensemble prediction দেয়, overfitting কমায়।
- SVR() – Support Vector Machine regression, sensitive to feature scaling।
  ```python 
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    
    # Define regression models
    regression_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Random Forest Regression': RandomForestRegressor(random_state=42),
    'Support Vector Regression': SVR()
    }
  ```
- model.fit(X_train, y_train) → মডেল train করা
- model.predict(X_test) → টেস্ট ডেটায় prediction করা 
- mean_squared_error(y_test, y_pred) → prediction error measure করছে।
- regression_models.items() ব্যবহার করলে আমরা মডেলের নাম ও instance একসাথে loop করতে পারি।

  ```python 
    from sklearn.metrics import mean_squared_error

    # mse = mean_squared_error(y_test, y_pred)

    for model in regression_models.values():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'{model}: {mse}')
  ```