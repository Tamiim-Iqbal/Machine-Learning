<h1 align="center">Model Enhancement</h1>

1️⃣ Goal
- **Model Enhancement** এর লক্ষ্য হলো মেশিন লার্নিং মডেলের প্রাথমিক পারফরম্যান্সের চেয়ে আরও ভালো এবং নির্ভরযোগ্য ফলাফল পাওয়া। এটি তিনটি প্রধান ধাপে অর্জিত হয়:
  - ফিচার ইঞ্জিনিয়ারিং (Feature Engineering): ইনপুট ডেটা থেকে আরও তথ্যবহুল এবং প্রাসঙ্গিক ফিচার তৈরি করা।
  - ক্লাস ইম্ব্যালান্স হ্যান্ডলিং (Class Imbalance Handling): ডেটার কমসংখ্যক ক্লাসগুলোর জন্য সমান শেখার সুযোগ নিশ্চিত করা।
  - হাইপারপ্যারামিটার অপটিমাইজেশন (Hyperparameter Optimization): মডেলের আচরণ এবং পারফরম্যান্স সর্বাধিক করার জন্য প্যারামিটার সমন্বয় করা।
  
### 2️⃣ Necessary Libraries
- General Libraries:
  ```python
    # General Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Preprocessing and Result Display
    from sklearn.model_selection import train_test_split
  ```

### 3️⃣ Importing a Dataset
- Google Drive মাউন্ট করা হয়েছে:
  ```python
    from google.colab import drive
    drive.mount('/content/drive') # এটি ব্যবহার করা হয় যাতে গুগল ড্রাইভ থেকে ফাইল (ডেটাসেট) সরাসরি লোড করা যায়।
- dataset load করার জন্য pandas এর read_csv() ফাংশন ব্যবহার করা।
  ```python
    df = pd.read_csv('/content/drive/MyDrive/path_to_your_file/garments_worker_productivity.csv')
    df.head()
    df.shape
  ```

### 5️⃣ Feature Set & Level Set (Target Variable)
- df থেকে Status এবং ID নামের কলামগুলো বাদ দেওয়া হচ্ছে (drop মানে মুছে ফেলা)। কারণ এসকল কলাম মডেল ট্রেনিং এর জন্য প্রয়োজনীয় নয়।
- drop() ফাংশন দিয়ে নির্দিষ্ট কলামগুলো ডিলিট করা হয়।
- axis=1 মানে কলাম (column) ডিলিট করা হচ্ছে, রো (row) নয়।
- X = ডেটাসেটের ইনপুট ফিচারগুলো (যা দিয়ে মডেল শিখবে)
- y = ডেটাসেটের আউটপুট বা টার্গেট ভ্যালু, যা তুমি প্রেডিক্ট করতে চাও।

  ```python 
    # Feature Set
    X = df.drop(['Status', 'ID'], axis=1)
    X.head()

    # Target Variable
    y = df['Status']
    y.head()
  ```

### 6️⃣ Train-Test Split
- Dataset কে 80% train এবং 20% test set এ ভাগ করা।
  ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### 7️⃣ Feature Engineering
- Numerical Features গুলোকে StandardScaler দিয়ে scale করা হচ্ছে যাতে মডেল ভালোভাবে শেখে।

### 7.1 Level Encoding 
- প্রথমে categorical ফিচারগুলো চিহ্নিত করতে হবে।
    ```python
    X_train.columns
    X_train.head(3)
    ```
- Only the categorical columns need label encoding. By looking at the data, we can understand that such columns are Drug, Sex, Ascites, Hepatomegaly, Spiders, Edema.
- Before starting label encoding, we will do these two things-
    - We will modify a copy of the dataframe instead of the original dataframe to keep the original values unchanged.
    - There are some NaN or missing values in the dataset. These values will remain as they are.
    - The same transformation should be applied to the test data. But for the test data we will use transform() instead of fit_transform() so that the encoding is same for both train and test sets.
  
- OrdinalEncoder ব্যবহার করা হয়েছে শব্দ/ক্যাটাগরিকে সংখ্যায় রূপান্তর করার জন্য।
- লিস্টে সেই সব কলামের নাম রাখা হয়েছে যেগুলোর মান লেখা (text) আকারে আছে। এগুলোকে সংখ্যায় এনকোড করা হবে।
- মূল ডেটা (X_train, X_test) যাতে বদলে না যায়, তাই এর কপি তৈরি করা হয়েছে।
- OrdinalEncoder() প্রতিটি কলামের টেক্সট ভ্যালুকে সংখ্যা (যেমন 1, 2, 3...) তে রূপান্তর করে।
- handle_unknown='use_encoded_value' মানে হলো, যদি টেস্ট ডেটায় নতুন কোনো মান পাওয়া যায় যেটা ট্রেন ডেটায় ছিল না,
তাহলে সেটাকে unknown_value=np.nan হিসেবে ধরবে (অর্থাৎ missing হিসেবে)।
- +1 দেওয়া হয়েছে যাতে এনকোডিং 0 থেকে না শুরু হয়ে 1 থেকে শুরু হয় (মানে প্রথম ক্যাটাগরি = 1, দ্বিতীয় = 2 ... )।
  ```python 
    from sklearn.preprocessing import OrdinalEncoder
    import numpy as np

    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

    # Create a copy to avoid modifying the original dataframe directly if needed later
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    for col in categorical_cols:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

        # Fit on train and transform both train & test
        X_train_encoded[col] = oe.fit_transform(X_train_encoded[[col]]) + 1
        X_test_encoded[col] = oe.transform(X_test_encoded[[col]]) + 1

    display(X_train_encoded.head())
  ```
- y_train / টার্গেট লেবেল (target labels) কেমন সেটা দেখা।
- এখানে আবারও OrdinalEncoder ব্যবহার করা হয়েছে, এবার টার্গেট ভ্যালুগুলো (যেমন “Yes”, “No” বা “Alive”, “Dead”)
সংখ্যায় রূপান্তর করার 
- y_train.values.reshape(-1, 1) মানে হলো y_train (যেটা এক কলামের Series) তাকে ২-ডি (column format) আকারে সাজানো, কারণ OrdinalEncoder ২-ডি ইনপুট চায়।
- এখন y_train এবং y_test উভয়ই numpy array হয়ে গেছে, যেখানে প্রতিটি লেবেল একটি সংখ্যা (যেমন 0, 1 ইত্যাদি)।
- ravel() ডেটাকে ফ্ল্যাট করে দেয় (মানে ২-ডি array → ১-ডি array)।
- তারপর numpy array থেকে আবার pandas Series বানানো হচ্ছে —
যাতে প্রয়োজনে pandas-এর ফিচার (index, display ইত্যাদি) ব্যবহার করা যায়।
  ```python 
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    y_train = oe.fit_transform(y_train.values.reshape(-1, 1))
    y_test = oe.transform(y_test.values.reshape(-1, 1))

    # Ravel will flatten the values
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Label encoding turns the single column into numpy array. We will retransform them into pandas dataframes. This is usually not necessary, but for some special purpose we will do this.
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    y_train
  ```

### 7.2 Removing missing values
- dropna() → যেসব রো-তে NaN (missing value) আছে, সেগুলো বাদ দেয়।
- reset_index(drop=True) → রো বাদ দেওয়ার পর নতুন ইনডেক্স তৈরি করে।
- y_train ও y_test এর ইনডেক্স X_train ও X_test এর সাথে মিলিয়ে নেওয়া হয়।
- Missing value বাদ দেওয়ার আগে ও পরে shape দেখে যাচাই করা হয় কত রো বাদ গেছে।
- এখন ডেটা পুরোপুরি clean, মডেল ট্রেন করার জন্য প্রস্তুত। 
  ```python
    # ডেটা চেক করা
    X_train_encoded.head()
    X_test_encoded.shape, y_test.shape

    print(f'Before removing missing values - X_train_encoded: {X_train_encoded.shape}')
    print(f'Before removing missing values - X_test_encoded: {X_test_encoded.shape}')

    # Missing value বাদ দেওয়া ও ইনডেক্স রিসেট করা
    X_train_encoded_dropped = X_train_encoded.dropna().reset_index(drop=True)
    y_train_dropped = y_train[X_train_encoded_dropped.index].reset_index(drop=True)
    X_test_encoded_dropped = X_test_encoded.dropna().reset_index(drop=True)
    y_test_dropped = y_test[X_test_encoded_dropped.index].reset_index(drop=True)

    print(f'After removing missing values: {X_train_encoded_dropped.shape}')
    print(f'After removing missing values: {X_test_encoded_dropped.shape}')
  ```

### Performance Check
- প্রয়োজনীয় লাইব্রেরি ইমপোর্ট
  ```python 
  # প্রয়োজনীয় লাইব্রেরি ইমপোর্ট
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
  ```
- মডেল ট্রেন এবং পারফরম্যান্স চেক করা
  ```python
    def train_and_evaluate_models(X_train, y_train, X_test, y_test):

        result = {}          # প্রতিটি মডেলের নাম ও accuracy সংরক্ষণের জন্য

        # ব্যবহৃত মেশিন লার্নিং মডেলগুলোর লিস্ট
        models = [
            LogisticRegression(max_iter=1000, random_state=42),
            RandomForestClassifier(random_state=42),
            AdaBoostClassifier(random_state=42),
            GradientBoostingClassifier(random_state=42),
            SVC(random_state=42),
            DecisionTreeClassifier(random_state=42),
            KNeighborsClassifier(),             # random_state নেই
            MLPClassifier(random_state=42),
            GaussianNB()                        # random_state নেই
    ]

    # প্রতিটি মডেল ট্রেন ও টেস্ট করা
    for model in models:
        model.fit(X_train, y_train)           # মডেল ট্রেন করা
        y_pred = model.predict(X_test)        # টেস্ট ডেটা প্রেডিক্ট করা

        # Accuracy স্কোর প্রিন্ট ও সংরক্ষণ
        print(f"{model.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred)}")
        result[model.__class__.__name__] = accuracy_score(y_test, y_pred)

    return result
  ```
- আগে train_and_evaluate_models() নামে একটি ফাংশন বানানো হয়েছিল।  যেটা বিভিন্ন মেশিন লার্নিং মডেল (Logistic Regression, Random Forest, SVM ইত্যাদি)
ট্রেন করে তাদের accuracy বের করে দেয়।
- এখন সেই ফাংশনটিকে প্রস্তুত করা clean ডেটা দিয়ে চালাচ্ছে।
  ```python 
    # Now call the function with your data
    result_encoded_dropped = train_and_evaluate_models(X_train_encoded_dropped, y_train_dropped, X_test_encoded_dropped, y_test_dropped)
  ```
### যদিও খুবই খারাপ পারফরম্যান্স পাওয়া গেছে। এখন আমরা ইমপ্রুভ করার চেষ্টা করব। 

### 7.3 Imputing Missing Values
- প্রতিটি কলামে কতগুলো missing value (NaN) আছে তা চেক করা
  ```python 
    X_train_encoded.isna().sum()
  ```
- SimpleImputer ব্যবহার করা হয় missing value পূরণ করার জন্য
- categorical_cols বাদ দিয়ে বাকি যেসব কলাম সংখ্যা ভিত্তিক (numerical), সেগুলো চিহ্নিত করা
- মূল ডেটা যাতে পরিবর্তন না হয়, তাই কপি তৈরি করা
- Missing value গুলো কলামের গড় মান (mean) দিয়ে পূরণ করা হবে।
- fit_transform() → ট্রেন ডেটার গড় বের করে সেটাই ট্রেনে বসায়।
- transform() → একই গড় মান টেস্ট ডেটাতেও ব্যবহার করে।
  ```python 
    from sklearn.impute import SimpleImputer

    # Identify numerical columns
    numerical_cols = X_train_encoded.columns.difference(categorical_cols)

    # Create copies of the original
    X_train_imputed = X_train_encoded.copy()
    X_test_imputed = X_test_encoded.copy()

    # Create an imputer object with mean strategy
    imputer = SimpleImputer(strategy='mean')

    # Apply imputation to numerical columns in train and test sets
    X_train_imputed[numerical_cols] = imputer.fit_transform(X_train_encoded[numerical_cols])
    X_test_imputed[numerical_cols] = imputer.transform(X_test_encoded[numerical_cols])

    X_train_imputed.head()
  ```
- ইম্পিউটেশনের আগে এবং পরে কতগুলো missing value ছিল সেটা দেখা 
  ```python 
    missing_before = X_train_encoded.isna().sum()
    missing_after = X_train_imputed.isna().sum()

    # Combine the missing value counts into a DataFrame
    missing_comparison = pd.DataFrame({
        'Before Imputation': missing_before,
        'After Imputation': missing_after
    })

    # Display the table, filtering to show only columns that had missing values
    display(missing_comparison[missing_comparison['Before Imputation'] > 0])
  ```
- Categorical কলামের Missing Value হ্যান্ডলিং : যেসব ক্যাটাগরিকাল কলামে missing value রয়ে গেছে, সেগুলো 0 দিয়ে পূরণ করা হয়েছে (simple approach)।
  ```python 
    X_train_imputed.fillna(0, inplace=True)
    X_test_imputed.fillna(0, inplace=True)

    X_train_imputed.isna().sum()
  ```
- Imputed ডেটা দিয়ে মডেল পারফরম্যান্স চেক করা
  ```python 
    result_imputed = train_and_evaluate_models(X_train_imputed, y_train, X_test_imputed, y_test)
  ```

### Model Accuracy Comparison (Dropped vs Imputed)
- কালার নির্বাচন colorblind-friendly এবং publication-friendly।
- models → মডেলের নামের লিস্ট।
- accuracies_dropped → missing value drop ডেটার accuracy।
- accuracies_imputed → missing value impute ডেটার accuracy
- তিটি মডেলের জন্য দুটি বার — একটি dropped, একটি imputed।
- Y-axis → Accuracy।
- X-axis → মডেলের নাম (45° ঘুরিয়ে readability বাড়ানো)।

  ```python 
    #@title Model Accuracy Comparison

    # Tableau 10 (colorblind-friendly and widely used in publications)
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf"   # Cyan
    ]

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    models = list(result_imputed.keys())
    accuracies_dropped = list(result_encoded_dropped.values())
    accuracies_imputed = list(result_imputed.values()) # Get accuracy for dropped, use 0 if model not present

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    rects2 = ax.bar(x - width/2, accuracies_dropped, width, label='Dropped Missing Values')
    rects1 = ax.bar(x + width/2, accuracies_imputed, width, label='Imputed Missing Values')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: Imputed vs. Dropped Missing Values')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()                 # লিজেন্ড দিয়ে বোঝানো হয়েছে কোন বার dropped এবং কোনটি imputed ডেটা।

    plt.tight_layout()          # চার্টের spacing ঠিক করে।
    plt.show()                  # চার্ট ডিসপ্লে করে।
  ```

### 7.4 Feature Scaling or Normalization
- StandardScaler (mean=0, std=1) ব্যবহার করা হয়েছে ডেটা স্কেলিং এর জন্য। 
    ```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Create a MinMaxScaler object
    # We could also use StandardScaler
    scaler = MinMaxScaler()

    # Apply scaling to the imputed data and create new variables
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_imputed.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_imputed.columns)

    X_train_scaled.head()
  ```
- স্কেলড ডেটা দিয়ে মডেল পারফরম্যান্স চেক করা
  ```python 
    result_scaled = train_and_evaluate_models(X_train_scaled, y_train, X_test_scaled, y_test)
  ```
### Model Accuracy Comparison (Dropped vs Imputed vs Scaled)
- ৩টি ডেটা সেটের মডেল পারফরম্যান্স তুলনা করা হইসে — dropped, imputed, এবং imputed + scaled।
    ```python
    #@title Model Accuracy Comparison

    models = list(result_scaled.keys())
    accuracies_dropped = [result_encoded_dropped.get(model, 0) for model in models] # Get accuracy for dropped, use 0 if model not present
    accuracies_imputed = [result_imputed.get(model, 0) for model in models] # Get accuracy for imputed, use 0 if model not present
    accuracies_imputed_scaled = list(result_scaled.values())

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 8))

    rects1 = ax.bar(x - width, accuracies_dropped, width, label='Dropped Missing Values')
    rects2 = ax.bar(x, accuracies_imputed, width, label='Imputed Missing Values')
    rects3 = ax.bar(x + width, accuracies_imputed_scaled, width, label='Imputed and Scaled Values')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: Dropped vs. Imputed vs. Imputed and Scaled Missing Values')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
  ```

### 7.5 One Hot Encoding
- OneHotEncoder ব্যবহার করা হয় categorical variables কে binary vector এ রূপান্তর করার জন্য।
    ```python 
    from sklearn.preprocessing import OneHotEncoder

    # Identify categorical columns (from earlier definition)
    categorical_cols = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

    # Create a OneHotEncoder object
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Apply one-hot encoding to the categorical columns in the scaled and imputed data
    X_train_ohe = ohe.fit_transform(X_train_scaled[categorical_cols])
    X_test_ohe = ohe.transform(X_test_scaled[categorical_cols])

    # Create DataFrames from the one-hot encoded arrays with appropriate column names
    # Get the feature names for the one-hot encoded columns
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    X_train_ohe_df = pd.DataFrame(X_train_ohe, columns=ohe_feature_names, index=X_train_scaled.index)
    X_test_ohe_df = pd.DataFrame(X_test_ohe, columns=ohe_feature_names, index=X_test_scaled.index)

    # Drop the original categorical columns from the scaled and imputed dataframes
    X_train_processed = X_train_scaled.drop(columns=categorical_cols)
    X_test_processed = X_test_scaled.drop(columns=categorical_cols)

    # Concatenate the one-hot encoded dataframes with the remaining numerical columns
    X_train_ohe_processed = pd.concat([X_train_processed, X_train_ohe_df], axis=1)
    X_test_ohe_processed = pd.concat([X_test_processed, X_test_ohe_df], axis=1)


    X_train_ohe_processed.head()
    X_train_ohe_processed.columns
  ```
- মডেল ট্রেন ও evaluation
  ```python 
  result_ohe = train_and_evaluate_models(X_train_ohe_processed, y_train, X_test_ohe_processed, y_test)
  ```

### Model Accuracy Comparison (Dropped vs Imputed vs Scaled vs One-Hot Encoded)
- ৪টি ডেটা সেটের মডেল পারফরম্যান্স তুলনা করা হইসে — dropped, imputed, imputed + scaled, এবং imputed + scaled + one-hot encoded।
    ```python
    #@title Model Accuracy Comparison

    models = list(result_ohe.keys()) # Use keys from ohe as it likely has all models
    accuracies_dropped = [result_encoded_dropped.get(model, 0) for model in models] # Get accuracy for dropped, use 0 if model not present
    accuracies_imputed = [result_imputed.get(model, 0) for model in models] # Get accuracy for imputed, use 0 if model not present
    accuracies_imputed_scaled = [result_scaled.get(model, 0) for model in models] # Get accuracy for imputed and scaled, use 0 if model not present
    accuracies_ohe = list(result_ohe.values())

    x = np.arange(len(models))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots(figsize=(20, 10))

    rects1 = ax.bar(x - width*2, accuracies_dropped, width, label='Dropped Missing Values')
    rects2 = ax.bar(x - width, accuracies_imputed, width, label='Imputed Missing Values')
    rects3 = ax.bar(x, accuracies_imputed_scaled, width, label='Imputed and Scaled Values')
    rects5 = ax.bar(x + width, accuracies_ohe, width, label='One-Hot Encoded, Imputed and Scaled Values')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: Different Preprocessing Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
  ```

### 7.6 Feature Selection
- Model-এর জন্য গুরুত্বপূর্ণ ফিচারগুলো বাছাই করা।
- Irrelevant বা noisy ফিচার বাদ দিলে:
    - মডেল দ্রুত ট্রেন হয়
    - Overfitting কম হয়
    - মডেল interpret করা সহজ হয়
- Feature selection করা যায়:
    - Correlation check করে
    - Recursive elimination (RFECV)
    - Model-based importance (Lasso, Random Forest)
- RFECV : Recursive Feature Elimination with Cross-Validation

  ```python 
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    log_reg = LogisticRegression(max_iter=1000, random_state=42)

    # Initialize RFECV with Logistic Regression and 5-fold cross-validation
    # Using StratifiedKFold to maintain class distribution in folds
    rfecv = RFECV(
        estimator=log_reg,       # Base model
        step=1,                  # একবারে ১টি feature বাদ দেবে
        cv=StratifiedKFold(5),   # 5-fold CV, class distribution বজায় রাখে
        scoring='accuracy'       # Performance metric
    )

    # Fit RFECV on training data
    rfecv.fit(X_train_ohe_processed, y_train)

    print(f"Optimal number of features : {rfecv.n_features_}")    # Model performance অনুযায়ী নির্বাচিত ফিচারের সংখ্যা।
    print("Selected features:")
    print(X_train_ohe_processed.columns[rfecv.support_])          # Boolean mask, কোন ফিচার retain করা হয়েছে।
  ```
### RFECV দিয়ে নির্বাচিত ফিচার ব্যবহার করে মডেল ট্রেন ও evaluation
- নিRFECV থেকে selected features বের করা
- ট্রেন ও টেস্ট ডেটা filter করে শুধুমাত্র selected features রাখা
- train_and_evaluate_models() দিয়ে সব মডেল ট্রেন ও accuracy পরীক্ষা করা
- **কম সংখ্যক, কিন্তু গুরুত্বপূর্ণ ফিচার ব্যবহার → মডেল দ্রুত ট্রেন হয়, overfitting কমে।**
  
  ```python 
    # Get the selected features
    selected_features = X_train_ohe_processed.columns[rfecv.support_]

    # Filter the training and testing data to include only the selected features
    X_train_selected = X_train_ohe_processed[selected_features]
    X_test_selected = X_test_ohe_processed[selected_features]

    # Train and evaluate models using the selected features
    result_selected_features = train_and_evaluate_models(X_train_selected, y_train, X_test_selected, y_test)
  ```

### Model Accuracy Comparison (All Preprocessing Features vs Selected Features)
- সব ফিচার ব্যবহার করে মডেল পারফরম্যান্স এবং শুধুমাত্র নির্বাচিত ফিচার ব্যবহার করে মডেল পারফরম্যান্স তুলনা করা হয়েছে।
  ```python 
    #@title Model Accuracy Comparison

    models = list(result_ohe.keys()) # Use keys from ohe as it likely has all models
    accuracies_dropped = [result_encoded_dropped.get(model, 0) for model in models] # Get accuracy for dropped, use 0 if model not present
    accuracies_imputed = [result_imputed.get(model, 0) for model in models] # Get accuracy for imputed, use 0 if model not present
    accuracies_imputed_scaled = [result_scaled.get(model, 0) for model in models] # Get accuracy for imputed and scaled, use 0 if model not present
    accuracies_ohe = list(result_ohe.values())
    accuracies_selected = [result_selected_features.get(model, 0) for model in models] # Get accuracy for selected features, use 0 if model not present


    x = np.arange(len(models))  # the label locations
    width = 0.12  # the width of the bars

    fig, ax = plt.subplots(figsize=(22, 10))

    rects1 = ax.bar(x - width*2.5, accuracies_dropped, width, label='Dropped Missing Values')
    rects2 = ax.bar(x - width*1.5, accuracies_imputed, width, label='Imputed Missing Values')
    rects3 = ax.bar(x - width*0.5, accuracies_imputed_scaled, width, label='Imputed and Scaled Values')
    rects5 = ax.bar(x + width*0.5, accuracies_ohe, width, label='One-Hot Encoded, Imputed and Scaled Values')
    rects6 = ax.bar(x + width*1.5, accuracies_selected, width, label='Selected Features (OHE, Imputed, Scaled)')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: Different Preprocessing Methods and Feature Selection')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
  ```
### 8️⃣ Imbalance Removal
- Dataset-এ প্রতিটি class-এর sample count দেখা ও Pie chart দিয়ে visualisation:
  ```python
    y_train.value_counts()

    #@title Pie Chart of Class Distribution
    plt.figure(figsize=(8, 8))
    y_train.value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution in Training Data')
    plt.ylabel('') # Remove default y-label
    plt.show()
  ```
- Pie chart থেকে বোঝা যায় কোন class কতটা underrepresented।
- এই dataset-এ class 1-এর sample সংখ্যা খুব কম।
- **SMOTE (Synthetic Minority Oversampling Technique):**
    - Underrepresented class-এর synthetic samples তৈরি করে।
    - Random duplication নয়, নতুন feature combination তৈরি করে।
- এখানে শুধুমাত্র class 1.0 oversample করা হচ্ছে।
- Synthetic rows তৈরি করে underrepresented class-এর সংখ্যা বাড়ানো হয়।
- Oversampling-এর পর training dataset বড় হয়ে যায়। Oversampling-এর ফলে minority class-এর সংখ্যা বাড়ে।
    ```python
    from imblearn.over_sampling import SMOTE
    import pandas as pd

    # Define target counts: only oversample class 1.0 up to 100 samples (instead of balancing fully)
    target_counts = {0.0: 188, 2.0: 125, 1.0: 80}

    # Apply SMOTE with custom sampling_strategy
    smote = SMOTE(sampling_strategy=target_counts, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Shape of training data before SMOTE:", X_train_scaled.shape)
    print("Shape of training data after SMOTE:", X_train_resampled.shape)

    print("\nClass distribution before SMOTE:")
    print(pd.Series(y_train).value_counts())

    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())
  ```
- মডেল ট্রেন ও evaluation
  ```python 
    result_resampled = train_and_evaluate_models(X_train_resampled, y_train_resampled, X_test_scaled, y_test)
  ```
### Model Accuracy Comparison (Before vs After Imbalance Removal)
- Imbalance removal-এর আগে ও পরে মডেল পারফরম্যান্স তুলনা করা
  ```python 
  #@title Model Accuracy Comparison

    models = list(result_ohe.keys()) # Use keys from ohe as it likely has all models
    accuracies_dropped = [result_encoded_dropped.get(model, 0) for model in models] # Get accuracy for dropped, use 0 if model not present
    accuracies_imputed = [result_imputed.get(model, 0) for model in models] # Get accuracy for imputed, use 0 if model not present
    accuracies_imputed_scaled = [result_scaled.get(model, 0) for model in models] # Get accuracy for imputed and scaled, use 0 if model not present
    accuracies_ohe = list(result_ohe.values())
    accuracies_selected = [result_selected_features.get(model, 0) for model in models] # Get accuracy for selected features, use 0 if model not present
    accuracies_resampled = [result_resampled.get(model, 0) for model in models] # Get accuracy for resampled, use 0 if model not present


    x = np.arange(len(models))  # the label locations
    width = 0.12  # the width of the bars

    fig, ax = plt.subplots(figsize=(22, 10))

    rects1 = ax.bar(x - width*2.5, accuracies_dropped, width, label='Dropped Missing Values')
    rects2 = ax.bar(x - width*1.5, accuracies_imputed, width, label='Imputed Missing Values')
    rects3 = ax.bar(x - width*0.5, accuracies_imputed_scaled, width, label='Imputed and Scaled Values')
    rects5 = ax.bar(x + width*0.5, accuracies_ohe, width, label='One-Hot Encoded, Imputed and Scaled Values')
    rects6 = ax.bar(x + width*1.5, accuracies_selected, width, label='Selected Features (OHE, Imputed, Scaled)')
    rects7 = ax.bar(x + width*2.5, accuracies_resampled, width, label='SMOTE Oversampled, Imputed and Scaled Values')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison: Different Preprocessing Methods and Feature Selection')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()
  ```
### 9️⃣ Hyperparameter Tuning
- Hyperparameters → মডেলের আগেই নির্ধারিত সেটিংস যা ডেটা থেকে শেখা হয় না।
    - যেমন: n_estimators, max_depth, learning_rate ইত্যাদি
- সঠিক tuning → মডেলের performance অনেক বাড়াতে পারে
- ভুল tuning → ভালো মডেলও খারাপ কাজ করতে পারে
- মডেলের hyperparameter গুলো optimize করা হয় যাতে মডেলের পারফরম্যান্স উন্নত হয়।
- GridSearchCV ব্যবহার করা হয় hyperparameter tuning এর জন্য।
- **GridSearchCV** → সব সম্ভাব্য parameter combination পরীক্ষা করে cross-validation score দেখে best combination বের করে।
- এখানে RandomForestClassifier মডেলের hyperparameter গুলো tune করা হয়েছে।
  ```python 
    from sklearn.model_selection import GridSearchCV

    # Random Forest-এর parameter grid
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }

    # GridSearchCV initialization
    rf_clf = RandomForestClassifier(random_state=42)

    grid_search_rf = GridSearchCV(
        estimator=rf_clf,
        param_grid=param_grid_rf,
        cv=StratifiedKFold(5),            # 5-fold cross-validation
        scoring='accuracy',               # performance metric
        n_jobs=-1,                        # সব CPU core ব্যবহার
        verbose=2                         # প্রগ্রেস মেসেজ দেখাবে
    )

    #Oversampled training data ব্যবহার (সব parameter combination পরীক্ষা করে best parameter নির্বাচন হবে)
    grid_search_rf.fit(X_train_resampled, y_train_resampled)

    print("Best parameters found: ", grid_search_rf.best_params_)             # Best hyperparameter combination
    print("Best cross-validation accuracy: ", grid_search_rf.best_score_)     # Best cross-validation accuracy

    # Test set evaluation
    best_rf_clf = grid_search_rf.best_estimator_
    y_pred_tuned_rf = best_rf_clf.predict(X_test_scaled)
    test_accuracy_tuned_rf = accuracy_score(y_test, y_pred_tuned_rf)
    print(f"Test set accuracy: {test_accuracy_tuned_rf:.4f}")
  ```
### Random Forest Accuracy Comparison (Before vs After Hyperparameter Tuning)
- Hyperparameter tuning-এর আগে ও পরে Random Forest মডেলের পারফরম্যান্স তুলনা করা হয়েছে।
    ```python
    #@title Random Forest Accuracy Comparison
    rf_accuracies = {
        'Dropped Missing Values': result_encoded_dropped.get('RandomForestClassifier', 0),
        'Imputed Missing Values': result_imputed.get('RandomForestClassifier', 0),
        'Imputed and Scaled Values': result_scaled.get('RandomForestClassifier', 0),
        'One-Hot Encoded, Imputed and Scaled Values': result_ohe.get('RandomForestClassifier', 0),
        'Selected Features (OHE, Imputed, Scaled)': result_selected_features.get('RandomForestClassifier', 0),
        'SMOTE Oversampled, Imputed and Scaled Values': result_resampled.get('RandomForestClassifier', 0),
        'Tuned (SMOTE, Scaled)': test_accuracy_tuned_rf
    }

    labels = list(rf_accuracies.keys())
    accuracies = list(rf_accuracies.values())

    # Tableau 10 colors
    colors = [
        "#1f77b4",  "#ff7f0e",  "#2ca02c",  "#d62728",
        "#9467bd",  "#8c564b",  "#e377c2"
    ]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, accuracies, color=colors)
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy Comparison Across Preprocessing Steps')
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0.75, 0.9)  # Zoom in on the relevant range for better visibility

    # Annotate bars with percentages
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.002,
            f"{height:.2%}",  # format as percentage
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.show()
  ```

