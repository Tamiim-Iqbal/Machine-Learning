<h1 align="center"> Classification 01 — Full Topic </h1>

### 1️⃣ Introduction 
- Purpose: বিভিন্ন machine learning classification মডেল শিখা এবং dataset নিয়ে কাজ করা।
- Overview: Classification হল এমন task যেখানে আমাদের output categorical হয়।
- Common Models: Logistic Regression, Decision Tree, Random Forest, KNN, SVM।

### 2️⃣ Libraries 
- লাইব্রেরি 
    ```python
    import pandas as pd             # ডেটা manipulation এর জন্য 
    ```
### 3️⃣ Load Data File
- dataset load করার জন্য pandas এর read_csv() ফাংশন ব্যবহার করা।
    ```python
    df = pd.read_csv('zoo.csv')
    df.head()
    ```

### 4️⃣ Creating Feature and Class Set
- features list-এ যেসব column আছে, সেগুলোকে input features হিসেবে ব্যবহার করা।
- X = df[features] দিয়ে dataframe থেকে শুধু এই column গুলোকে আলাদা করা।
- X হলো feature matrix, যা মডেল ট্রেনিং-এর input হিসেবে ব্যবহার হবে।
    ```python
    features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 
            'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
            'fins', 'legs', 'tail', 'domestic', 'catsize']
    X = df[features]
    X                # feature matrix
    ```

- **type** column কে target variable হিসেবে নির্বাচন করা।
- y হলো সেই column-এর values, যা মডেল predict করবে।
- y শুধুমাত্র labels বা output ধারণ করে, input features নয়।
    ```python
    y = df['type']
    y
   ```

### 5️⃣ Train-Test Split 
- লাইব্রেরি :
  ```python
    from sklearn.model_selection import train_test_split
    ```
- Dataset কে 80% train এবং 20% test set এ ভাগ করা।
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
    ```
### আমাদের সব লাইব্রেরি, ডাটাসেট, ট্রেনিং ও টেস্টিং ডাটা নেয়া শেষ। এখন আমরা মডেল ট্রেনিং করাবো।


### 6️⃣ Classification
- Most of the classification tasks we will perform consist of the following basic steps
  1. Import the necessary library or class or method
  2. Build or create model
  3. Train model
  4.  Test model

### 6.1 Decision Tree 
- Decision Tree train এবং predict করা।
- লাইব্রেরি :
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    model_dTree = DecisionTreeClassifier() 
    ```
- Build or Create Model
    ```python
    hist_dTree = model_dTree.fit(X_train, y_train)
    ```
- See the prediction for an animal of your choice
  ```python
    random_animal = pd.DataFrame([[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 4, 1, 0, 0]])
    model_dTree.predict(random_animal)
    ```
- Get prediction of the model for test dataset
    ```python
    result_dTree = model_dTree.predict(X_test)
    ```
- Get Classification accuray, precision, etc.
  ```python
  print(classification_report(y_test, result_dTree))
  ```

### 6.2 Logistic Regression
- Train logistic regression and predict on test set.
  ```python
  from sklearn.linear_model import LogisticRegression
  
  # Build or Create Model
  model_logReg = LogisticRegression()

  # Train Model
  hist_logReg = model_logReg.fit(X_train, y_train)

  # Get prediction of the model for test dataset
  result_logReg = model_logReg.predict(X_test)

  # Evaluate
  print(classification_report(y_test, result_logReg))
  ```


### 6.3 Naive Bayes
- সম্পূর্ণ কোড :
    ```python
    from sklearn.naive_bayes import GaussianNB
    
    # Initialize Gaussian Naive Bayes
    gnb = GaussianNB()

    # Train and predict
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    # Evaluate
    print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
    print(classification_report(y_test, y_pred))
    ```
### 6.4 Support Vector Machine
- সম্পূর্ণ কোড :
  ```python
    from sklearn.svm import SVC
    # Initialize SVM (default is RBF kernel)
    model_svm = SVC(kernel='rbf', gamma='scale')

    # Train the model
    model_svm.fit(X_train, y_train)

    # Predict on test data
    y_pred_svm = model_svm.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred_svm))
  ```

### 6.5 Random Forest
- সম্পূর্ণ কোড :
  ```python
  from sklearn.ensemble import RandomForestClassifier

  # Initialize Random Forest
  model_rf = RandomForestClassifier()

  # Train the model
  model_rf.fit(X_train, y_train) 

  # Predict on test data
  result_rf = model_rf.predict(X_test)

  # Evaluate
  print(classification_report(y_test, result_rf))
  ```

### 7️⃣ Related Topic to Learn
- Confusion Matrix
- Visualization