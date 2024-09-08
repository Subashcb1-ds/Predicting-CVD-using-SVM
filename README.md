 # **Predicting Cardiovascular Disease Using Support Vector Machines (SVM)**

## **Project Overview**
Cardiovascular disease (CVD) is a leading cause of death globally, and early identification of risk factors is crucial for prevention. In this project, I utilized the **Support Vector Machine (SVM)** algorithm with an **RBF kernel** to predict the risk of cardiovascular disease. The dataset used is the **Framingham Heart Study** dataset, a well-known source for CVD prediction research.

The goal is to develop a machine learning model that can predict the likelihood of a patient developing CVD based on their health characteristics.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building: Support Vector Machine (SVM)](#model-building-support-vector-machine-svm)
5. [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Contact](#contact)

---

## **Introduction**
The main objective of this project is to predict cardiovascular disease (CVD) in individuals using key health factors like **age**, **blood pressure**, **cholesterol levels**, and **smoking habits**. The **Support Vector Machine (SVM)** model is well-suited for binary classification problems like this and is used here to classify individuals as either at risk for CVD or not.

---

## **Dataset**
The dataset used is derived from the **Framingham Heart Study** and contains several features related to cardiovascular health. The primary features and the target variable are:

- **Features**:
  - `age`: Age of the individual.
  - `sysBP`: Systolic blood pressure.
  - `totChol`: Total cholesterol.
  - `currentSmoker`: Whether the individual is a current smoker (binary).
  - `glucose`: Glucose level.
  - `BMI`: Body Mass Index.
  - `prevalentHyp`: Whether the individual has hypertension (binary).
  - `diabetes`: Whether the individual has diabetes (binary).
  - `cigsPerDay`: Number of cigarettes smoked per day.
  - `diaBP`: Diastolic blood pressure.

- **Target**:
  - `TenYearCHD`: Whether the individual has the 10-year risk of developing coronary heart disease (binary).

---

## **Data Preprocessing**
Before fitting the SVM model, the dataset was preprocessed to ensure data quality and improve model performance.

### **Steps Involved:**
1. **Handling Missing Values**: Missing values were handled using appropriate techniques, such as imputing the mean for continuous features.
   ```python
   df_filled = df.fillna(df.mean())
   ```
   
2. **Feature Selection**: Only relevant features (like age, blood pressure, cholesterol, etc.) were selected for model training.
   
3. **Train-Test Split**: The dataset was split into training (80%) and testing (20%) sets.
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

4. **Scaling the Data**: Feature scaling was applied using **StandardScaler** to ensure all features are on a similar scale, which is important for SVM.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```

---

## **Model Building: Support Vector Machine (SVM)**

### **SVM Model with RBF Kernel**
An **SVM** with a **Radial Basis Function (RBF)** kernel was used for this classification problem. The RBF kernel is effective in non-linear classification tasks, and it maps the input space into higher dimensions to find an optimal hyperplane for separation.

1. **Initializing the Model**:
   ```python
   from sklearn.svm import SVC
   svm_model = SVC(kernel='rbf', probability=True, random_state=42)
   ```

2. **Training the Model**: The model was trained using the training dataset.
   ```python
   svm_model.fit(X_train_scaled, y_train)
   ```

3. **Making Predictions**: Predictions were made on the test dataset.
   ```python
   y_pred = svm_model.predict(X_test_scaled)
   y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
   ```

---

## **Model Evaluation**
After training, the model was evaluated using various metrics to assess its performance:

1. **Accuracy**: The overall accuracy of the model was computed, which measures how many correct predictions the model made.
   ```python
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {accuracy:.2f}')
   ```

2. **Confusion Matrix**: The confusion matrix helps visualize the number of true positives, true negatives, false positives, and false negatives.
   ```python
   from sklearn.metrics import confusion_matrix
   import seaborn as sns
   conf_matrix = confusion_matrix(y_test, y_pred)
   sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
   ```

3. **Classification Report**: This report includes precision, recall, and F1-score, providing a detailed overview of the model’s performance.
   ```python
   from sklearn.metrics import classification_report
   print(classification_report(y_test, y_pred))
   ```

4. **ROC-AUC Score**: The ROC-AUC score was calculated to evaluate how well the model distinguishes between positive and negative classes.
   ```python
   from sklearn.metrics import roc_auc_score
   roc_auc = roc_auc_score(y_test, y_pred_proba)
   print(f'ROC-AUC Score: {roc_auc:.2f}')
   ```

5. **ROC Curve**: The ROC curve visualizes the model's ability to differentiate between the positive and negative classes across different thresholds.
   ```python
   from sklearn.metrics import roc_curve
   fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
   plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
   plt.plot([0, 1], [0, 1], linestyle='--')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.legend()
   plt.show()
   ```

---

## **Results**
- **Accuracy**: The model achieved an accuracy of **0.86**.
- **ROC-AUC Score**: The ROC-AUC score was **0.60**, indicating the model’s capability in distinguishing between patients with and without CVD.

---

## **Conclusion**
The SVM model with an RBF kernel effectively predicted the risk of cardiovascular disease. The results indicate that **age**, **blood pressure**, **cholesterol**, and **smoking habits** are significant contributors to CVD risk.

---

## **Contact**
If you have any questions, feel free to reach out:

- **GitHub**: [Subashcb1-ds](https://github.com/Subashcb1-ds)
- **Email**: subashchandrabosem524@gmail.com
