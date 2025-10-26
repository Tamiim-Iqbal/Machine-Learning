<h1 align="center"> Classification 02 — Classification by Extracting Features  </h1>

### 1️⃣ Goal
- টেক্সট ডেটা থেকে ফিচার এক্সট্রাকশন (Feature Extraction) শেখা এবং সেটির মাধ্যমে টেক্সট ক্লাসিফিকেশন মডেল তৈরি করা।
- ডেটাসেট হিসেবে ব্যবহৃত হয়েছে BBC news dataset, যেখানে বিভিন্ন নিউজ ক্যাটেগরি আছে যেমন — business, politics, sports, tech, entertainment ইত্যাদি।

### 2️⃣ Importing Necessary Libraries
- সব গুলো লাইব্রেরি :
    ```python
    # ডেটা হ্যান্ডলিং ও প্রসেসিং-এর জন্য
    import pandas as pdimport pandas as pd
    import numpy as 
    
    # ডেটা ভিজুয়ালাইজেশনের জন্য
    import matplotlib.pyplot as plt
    import seaborn as 
    
    # ডেটা ট্রেনিং ও টেস্ট ভাগ করার জন্য
    from sklearn.model_selection import train_test_split

    # টেক্সট ফিচার তৈরি করার জন্য
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    # ক্লাসিফিকেশন মডেল
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    # মডেল ইভ্যালুয়েশনের জন্য
    from sklearn.metrics import classification_report, confusion_matrix
    ```

### 3️⃣ Importing a Dataset
- Google Drive মাউন্ট করা হয়েছে: 
  ```python
  from google.colab import drive
  drive.mount('/content/drive') # এটি ব্যবহার করা হয় যাতে গুগল ড্রাইভ থেকে ফাইল (ডেটাসেট) সরাসরি লোড করা যায়।
  ```
- Excel file (BBC ডেটাসেট) লোড করা হয়েছে:
  ```python
  df = pd.read_excel("/content/drive/MyDrive/path_to_your_file/bbc_dataset.xlsx")
  df.head()
  ```
### 4️⃣ Exploring the Dataset & Class Distribution
- মোট কতগুলো ডেটা আছে এবং কয়টি ক্লাস আছে তা প্রিন্ট করা হয়।
- sns.countplot() দিয়ে প্রতিটি ক্লাসে কতটি ডেটা আছে তা দেখানো হয় (যেমন sports, politics ইত্যাদি)।
  ```python
    print("Number of samples:", len(df))          # মোট কতগুলো রো আছে
    print("Classes:", df['label'].unique())       # ইউনিক ক্লাসগুলো দেখা / মোট কয়টা ক্লাস আছে

    # প্রতিটি ক্লাসে কয়টা ডেটা আছে তার গ্রাফ
    sns.countplot(x="label", data=df)             
    plt.title("Class Distribution")               # টাইটেল সেট করা
    plt.show()                                    # গ্রাফ দেখানো


### 5️⃣ Text Cleaning (Preprocessing)
- nltk stopwords ব্যবহার করে অপ্রয়োজনীয় শব্দ বাদ দেওয়া হয়, যেমন — "is", "the", "and" ইত্যাদি।
- re.sub() দিয়ে বিশেষ চিহ্ন, সংখ্যা ইত্যাদি বাদ দেওয়া হয়।
- সব টেক্সটকে ছোট হাতের অক্ষরে (lowercase) রূপান্তর করা হয়।
  ```python 
    # লাইব্রেরি ইমপোর্ট:
    import re                                      # রেগুলার এক্সপ্রেশন ব্যবহার করে টেক্সট পরিষ্কার করতে
    from nltk.corpus import stopwords              # অপ্রয়োজনীয় শব্দ বাদ দিতে
    import nltk                                    # Natural Language Toolkit
    nltk.download("stopwords")                     # stopwords ডাউনলোড করা

    stop_words = set(stopwords.words("english"))   # ইংরেজি stopwords সেটে রাখা

    # টেক্সট পরিষ্কার করার ফাংশন
    def clean_text(text):
        text = text.lower()                                  # সব অক্ষর ছোট হাতের করা
        text = re.sub(r"[^a-z\\s]", "", text)                # অক্ষর ছাড়া সব কিছু বাদ দেওয়া
        return " ".join([w for w in text.split() if w not in stop_words])  # stopword (অপ্রয়োজনীয় শব্দ) বাদ দেয়া 

    # ডেটাফ্রেমে নতুন কলাম তৈরি করা যেখানে ক্লিন টেক্সট থাকবে
    df["clean_text"] = df["text"].apply(clean_text)
    ```

### 6️⃣ Clean Data Preview
- ক্লিন করা টেক্সটের কিছু উদাহরণ দেখা:
  ```python
    df.head()    # এখন “clean_text” কলাম দেখা যাবে যেখানে প্রসেস করা টেক্সট আছে
    df[["text", "clean_text"]].head()  # আসল টেক্সট এবং ক্লিন টেক্সটের প্রথম ৫টি উদাহরণ দেখানো
    ```
### 7️⃣ Feature Extraction (Keyword Feature Extraction)
- Keyword Feature Extraction হলো এক ধরণের ম্যানুয়াল ফিচার ইঞ্জিনিয়ারিং,
যেখানে আমরা কিছু নির্দিষ্ট ক্যাটেগরির গুরুত্বপূর্ণ শব্দ (keywords) বেছে নিয়ে
তাদের ফ্রিকোয়েন্সি (বারবার উপস্থিতি) গুণি।

    ```python
        # প্রতিটি ক্যাটেগরির জন্য গুরুত্বপূর্ণ কীওয়ার্ডের তালিকা
        keywords = [
        # Business
        "market", "stock", "bank", "economy", "finance", "trade", "growth",
        # Politics
        "election", "government", "minister", "policy", "parliament", "law",
        # Sport
        "football", "cricket", "tennis", "match", "team", "goal", "tournament",
        # Tech
        "technology", "computer", "internet", "software", "ai", "digital", "innovation",
        # Entertainment
            "film", "music", "movie", "actor", "actress", "award", "theatre"
        ]

        # প্রতিটি টেক্সটে এই কীওয়ার্ডগুলো কতবার এসেছে তা গুনে ফিচার ভেক্টর তৈরি করে।
        def extract_keyword_features(texts, keywords):
        features = []                             # ফিচার রাখার জন্য খালি লিস্ট
        for text in texts:                        # প্রতিটি টেক্সটের জন্য
            words = text.split()                  # শব্দে ভাগ করা
            counts = [words.count(kw) for kw in keywords]  # প্রতিটি কীওয়ার্ড কয়বার এসেছে তা গণনা করা
            features.append(counts)               # রেজাল্ট ফিচারে যোগ করা
        return np.array(features)                 # NumPy অ্যারে হিসেবে রিটার্ন করা

        # ট্রেন ও টেস্ট সেটে ফিচার এক্সট্রাকশন চালানো
        X_train_keywords = extract_keyword_features(X_train, keywords)
        X_test_keywords = extract_keyword_features(X_test, keywords)

        print("Keyword feature shape:", X_train_keywords.   shape)  # আকার দেখা (নমুনা সংখ্যা × কীওয়ার্ড সংখ্যা)
    ```

### এখান পর্যন্ত ডেটা লোড → ক্লিনিং → ফিচার এক্সট্রাকশন সম্পূর্ণ হয়েছে।

### 8️⃣ Keyword Frequency by Label
- প্রতিটি লেবেল (যেমন — business, politics, sport) অনুযায়ী
 নির্দিষ্ট করা keywords গুলোর ফ্রিকোয়েন্সি গণনা করে টেবিল আকারে দেখানো।
  ```python
    # লাইব্রেরি:
    from collections import Counter

    # প্রতিটি ইউনিক label (যেমন — business, sports, politics ইত্যাদি) এর জন্য একটা Counter() অবজেক্ট তৈরি করা হয়েছে। 
    # এই Counter দিয়ে আমরা ঐ লেবেলের অধীনে থাকা শব্দগুলোর ফ্রিকোয়েন্সি রাখব।
    def keyword_frequency_by_label(df, keywords):
        label_keyword_counts = {label: Counter() for label in df["label"].unique()}
        
    # Count keyword occurrences per label : iterrows() দিয়ে DataFrame-এর প্রতিটি রো ঘুরে দেখা হচ্ছে। clean_text কলাম থেকে প্রতিটি শব্দের সংখ্যা গোনা হচ্ছে। তারপর যদি শব্দটি আমাদের keywords list-এর মধ্যে থাকে, তবে সেই শব্দের count ঐ লেবেলের Counter-এ যোগ হচ্ছে।
        for _, row in df.iterrows():
            words = row["clean_text"].split()
            counts = Counter(words)
            for kw in keywords:
                if kw in counts:
                    label_keyword_counts[row["label"]][kw] += counts[kw]

    # Convert to DataFrame : সব Counter dictionary-কে একত্রে pandas DataFrame-এ রূপান্তর করা হয়। .fillna(0) → যদি কোনো শব্দ কোনো ক্যাটেগরিতে না থাকে, সেখানে 0 বসানো হয়। .astype(int) → সব মান integer-এ রূপান্তরিত হয়।
        freq_df = pd.DataFrame(label_keyword_counts).fillna(0).astype(int)
        return freq_df

    # Run on your dataset : df (BBC ডেটাসেট) ও keywords ইনপুট হিসেবে দিচ্ছে
    freq_df = keyword_frequency_by_label(df, keywords)

    # Show top keyword frequencies per label : DataFrame-এর index (অর্থাৎ শব্দ) গুলো alphabetically sort করা হচ্ছে। তারপর প্রিন্ট করে দেখাচ্ছে, কোন লেবেলে কোন শব্দ কতবার এসেছে।
    print(freq_df.sort_index())
  ```
### 9️⃣ Keyword Frequency Visualization (Heatmap + Bar Chart)
- sns.heatmap() প্রতিটি কীওয়ার্ড বনাম প্রতিটি লেবেলের ফ্রিকোয়েন্সি টেবিলকে রঙিনভাবে দেখায়। যে জায়গায় মান বেশি, সেটি গাঢ় রঙে দেখা যাবে।
  ```python 
    # 🔥 Heatmap of keyword frequencies
    plt.figure(figsize=(12,8))                                # ফিগারের সাইজ সেট করা
    sns.heatmap(freq_df, annot=True, fmt="d", cmap="YlGnBu")  # প্রতিটি সেলে সংখ্যা সহ হিটম্যাপ আঁকা
    plt.title("Keyword Frequency per Label")                  # গ্রাফের টাইটেল
    plt.xlabel("News Label")                                  # এক্স-অক্ষের নাম
    plt.ylabel("Keyword")                                     # ওয়াই-অক্ষের নাম
    plt.show()                                                # গ্রাফ প্রদর্শন
  ``` 
- প্রতিটি লেবেলের জন্য আলাদা বার চার্ট তৈরি হয়। এতে দেখা যায় কোন শব্দগুলো ঐ ক্যাটেগরির সবচেয়ে বেশি গুরুত্বপূর্ণ।
  ```python 
    # 📊 Bar chart: Top keywords per label
    for label in df["label"].unique():                   # প্রতিটি লেবেলের জন্য লুপ চালানো
        top_keywords = freq_df[label].sort_values(ascending=False).head(10)  # শীর্ষ ১০টি কীওয়ার্ড নির্বাচন
        plt.figure(figsize=(8,4))                        # প্রতিটি চার্টের সাইজ নির্ধারণ
        sns.barplot(x=top_keywords.values, y=top_keywords.index, palette="viridis")  # বার চার্ট তৈরি করা
        plt.title(f"Top Keywords in {label} Articles")   # প্রতিটি লেবেলের টাইটেল
        plt.xlabel("Frequency")                          # এক্স-অক্ষের নাম
        plt.ylabel("Keyword")                            # ওয়াই-অক্ষের নাম
        plt.show()                                       # চার্ট দেখানো
  ```

### 1️⃣0️⃣ Combine BoW + TF-IDF + Keyword Features (Hybrid Feature Extraction)
- BoW (Bag of Words) : প্রতিটি শব্দের উপস্থিতির সংখ্যা (frequency) ধরে রাখে।
- TF-IDF : গুরুত্বপূর্ণ শব্দগুলোর ওজন (weight) বেশি দেয়, সাধারণ শব্দের ওজন কমায়।
- Keyword Features : ম্যানুয়ালি বাছাই করা ক্যাটেগরি নির্দিষ্ট শব্দগুলোকে ধরে রাখে।
- hstack() : তিনটি ফিচার ম্যাট্রিক্সকে পাশে পাশে জোড়া লাগায় (horizontal stack)।
- এখন একটি হাইব্রিড ফিচার সেট পাওয়া গেছে যেখানে প্রতিটি **ডকুমেন্টে = BoW + TF-IDF + Keyword** তথ্য একত্রে আছে।
এটি মডেলের **accuracy** বাড়াতে অনেক সময় সাহায্য করে কারণ এতে
**statistical + semantic + domain** তথ্য সব একসাথে থাকে।

  ```python 
    from scipy.sparse import hstack   # sparse ম্যাট্রিক্সগুলো একসাথে যুক্ত করার জন্য

    # Define vectorizers
    bow_vectorizer = CountVectorizer(max_features=5000)   # Bag of Words ফিচার (সর্বোচ্চ ৫০০০ শব্দ)
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # TF-IDF ফিচার (সর্বোচ্চ ৫০০০ শব্দ)

    # Fit and transform BoW
    X_train_bow = bow_vectorizer.fit_transform(X_train)   # ট্রেন ডেটা থেকে BoW ফিচার শেখা
    X_test_bow = bow_vectorizer.transform(X_test)         # টেস্ট ডেটা রূপান্তর করা

    # Fit and transform TF-IDF
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # ট্রেন ডেটা থেকে TF-IDF শেখা
    X_test_tfidf = tfidf_vectorizer.transform(X_test)        # টেস্ট ডেটা রূপান্তর করা

    # Combine all features together
    X_train_combined = hstack([X_train_bow, X_train_tfidf, X_train_keywords])  # তিনটি ফিচার একত্রে
    X_test_combined = hstack([X_test_bow, X_test_tfidf, X_test_keywords])

    print("✅ Final combined feature shape:", X_train_combined.shape)
  ```
### 1️⃣1️⃣ Train & Evaluate Multiple Models with Hybrid 
- Models dictionary: একসাথে অনেক মডেল সংরক্ষণ করা হচ্ছে।
- Training: প্রতিটি মডেলকে X_train_combined এবং y_train দিয়ে ট্রেন করা হচ্ছে।
- Prediction: হাইব্রিড ফিচারের ওপর ভিত্তি করে টেস্ট ডেটা থেকে লেবেল প্রেডিকশন।
- Evaluation: 
  - classification_report() → precision, recall, f1-score, support
  - confusion_matrix() → কোন লেবেল কতটা সঠিক বা ভুল হয়েছে
- Visualization: প্রতিটি মডেলের জন্য heatmap দেখানো হচ্ছে, যাতে সহজে পারফরম্যান্স তুলনা করা যায়।
  ```python 
    # Import models
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Define models dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    # Dictionary to store results
    results = {}

    # Train, predict & evaluate each model
    for name, model in models.items():
        print(f"\n Training {name}...")                  # কোন মডেল ট্রেনিং হচ্ছে তা প্রিন্ট করা
        model.fit(X_train_combined, y_train)             # ট্রেনিং ডেটা দিয়ে মডেল ফিট/ট্রেন করা
        preds = model.predict(X_test_combined)           # টেস্ট ডেটা থেকে প্রেডিকশন করা

        # Print classification report
        print(f"--- {name} Report ---")
        print(classification_report(y_test, preds))      # precision, recall, f1-score দেখানো

        # Store results
        results[name] = classification_report(y_test, preds, output_dict=True)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix - {name}")
        plt.show()
  ```
### 1️⃣2️⃣ Accuracy Comparison Bar Chart
- results dictionary-তে সব মডেলের classification report আছে।
- আমরা প্রতিটি মডেলের accuracy বের করেছি।
- sns.barplot() দিয়ে একটি বার চার্ট তৈরি করেছি যা মডেলের তুলনা সহজ করে।
- X-অক্ষ → মডেলের নাম
- Y-অক্ষ → Accuracy (%)
  ```python
    # Extract accuracy scores from results dictionary
    accuracy_scores = {name: results[name]["accuracy"] for name in results}

    # Plot bar chart
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.show()
  ```

### 1️⃣3️⃣ Training Logistic Regression (Hybrid Features)
- Hybrid Features: মডেল ট্রেন হচ্ছে BoW + TF-IDF + Keyword ফিচার ব্যবহার করে।
- max_iter=1000: লজিস্টিক রিগ্রেশনের জন্য iteration সংখ্যা।
- classification_report(): প্রতিটি ক্লাসের precision, recall, f1-score এবং overall accuracy দেখায়।
  ```python 
    # লাইব্রেরি ইমপোর্ট
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Initialize and train Logistic Regression model
    lr_model = LogisticRegression(max_iter=1000)     # সর্বাধিক 1000 iteration পর্যন্ত ট্রেনিং

    print("\nTraining Logistic Regression...")
    lr_model.fit(X_train_combined, y_train)          # ট্রেনিং ডেটা দিয়ে মডেল ফিট/ট্রেন করা
    lr_preds = lr_model.predict(X_test_combined)     # টেস্ট ডেটা থেকে লেবেল প্রেডিকশন করা

    print("--- Logistic Regression Report ---")
    print(classification_report(y_test, lr_preds))   # precision, recall, f1-score দেখানো
  ```
### 1️⃣4️⃣ Testing : Predict Category for New Article
- Preprocessing: নতুন আর্টিকেলকে ক্লিন করা (ছোট হাতের অক্ষরে রূপান্তর, স্পেশাল ক্যারেক্টার ও stopwords বাদ দেওয়া)
- Feature Extraction: নতুন আর্টিকেলের জন্য BoW, TF-IDF এবং Keyword ফিচার তৈরি করা
- Combine Features: hstack() দিয়ে সব ফিচার একত্রে রাখা
- Prediction: ট্রেইন্ড মডেল ব্যবহার করে লেবেল প্রেডিকশন। prediction[0] দিয়ে কেবল লেবেল রিটার্ন করছে। 
  ```python 
    # Predict Category for New Article
    def predict_news_category(article, model, bow_vectorizer, tfidf_vectorizer, keywords):
        import re
        import numpy as np

        # Preprocess text
        stop_words = set(stopwords.words("english"))     # stopwords সেট তৈরি
        def clean_text(text):
            text = text.lower()                          # ছোট হাতের অক্ষরে রূপান্তর
            text = re.sub(r"[^a-z\s]", "", text)         # স্পেশাল ক্যারেক্টার ও সংখ্যা বাদ দেওয়া    
            return " ".join([w for w in text.split() if w not in stop_words])             # stopwords বাদ দেওয়া

        clean_article = clean_text(article)              # আর্টিকেল ক্লিন করা

        # Feature extraction
        bow_feat = bow_vectorizer.transform([clean_article])      # BoW
        tfidf_feat = tfidf_vectorizer.transform([clean_article])  # TF-IDF

        # Keyword frequency
        def extract_keyword_features(texts, keywords):
            features = []
            for text in texts:
                words = text.split()
                counts = [words.count(kw) for kw in keywords]     # প্রতিটি কীওয়ার্ডের ফ্রিকোয়েন্সি গণনা
                features.append(counts)
            return np.array(features)

        keyword_feat = extract_keyword_features([clean_article], keywords)

        # Combine features
        from scipy.sparse import hstack
        combined_feat = hstack([bow_feat, tfidf_feat, keyword_feat])

        # Predict
        prediction = model.predict(combined_feat)           # প্রেডিকশন
        return prediction[0]                                # লেবেল রিটার্ন

    # Example: নতুন আর্টিকেল প্রেডিকশনExample usage:
    new_article = "Bangladesh has made notable progress in digital financial transactions but remains far from becoming a cashless economy, experts said yesterday, calling for stronger policy support, improved infrastructure, and wider adoption to reduce reliance on physical currency."
    predicted_label = predict_news_category(new_article, lr_model, bow_vectorizer, tfidf_vectorizer, keywords)
    print(f"Predicted News Category: {predicted_label}")   # প্রেডিক্টেড লেবেল প্রিন্ট করা
    ```
