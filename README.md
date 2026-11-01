# Steel Plate Fault Classification

## a. Problem statement
Steel plates may develop different types of surface defects, such as scratches, stains, bumps and pastry marks, which can affect the overall quality of the finished product. Identifying these defects at an early stage is important as delays in detection can lead to additional costs.Traditional manual inspection is time consuming and may also result in differences in judgement between inspectors.
The project aims to develop a machine learning based system that can identify and classify surface defects using 27 features recorded for each steel plate. These features include plate dimensions,defect location and areas, luminosity,steel type, thickness and shape related measurements.The system classifies defects into seven categories:Pastry, Z_Scratch, K_Scratch, K_Scratch,Stains,Dirtiness, Bumps,Other_Faults.
To identify the most suitable classification approach five machine learning algorithms- Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine-are trained and evaluated using the same dataset.Their performance is compared based on accuracy, AUC,Precision,recall,F1-Score and Matthews Correlation Coefficient(MCC). Finally the trained models are integrated into a streamlit based web application to make defect prediction easier and more accessible.
## b. Dataset description
- **Source:** UCI Machine Learning Repository, Steel Plates Faults dataset (https://archive.ics.uci.edu/dataset/198)
- **Size:** 1,941 instances and 27 input features, which meets the assignment minimums (12 features, 500 instances).
- **Features:** numeric measurements of each steel plate, such as its position (X and Y minimum and maximum), area in pixels, perimeters, luminosity values, steel type, plate thickness and several shape and edge indices.
- **Target:** the type of fault, with 7 classes: Other_Faults (673), Bumps (402), K_Scratch (391), Z_Scratch (190), Pastry (158), Stains (72) and Dirtiness (55). The classes are imbalanced, so a few fault types have far fewer examples than the rest.
- **Missing values:** none.
- **Preprocessing:** in the original data the 7 fault types come as separate 0/1 columns, so I combined them into a single label column and then label-encoded it. Features were standardised with StandardScaler inside a pipeline for Logistic Regression, KNN and SVM. Decision Tree and Naive Bayes were used without scaling.
- **Train/test split:** 80/20, stratified by fault type, random seed 20, giving 1,552 training rows and 389 test rows. The test split is saved as `test_data.csv` and used by the Streamlit app.
- **Model settings:** default scikit-learn settings (with `max_iter=2000` for Logistic Regression and `probability=True` for SVM).
## c. GitHub repository link
https://github.com/Tanmayee1999/Steel-Fault-Classifier-App

## d. Models used
All five models were trained on the same training split (1,552 rows) and evaluated on the same test split (389 rows):
1. **Logistic Regression**: StandardScaler + LogisticRegression (max_iter=2000)
2. **Decision Tree**: DecisionTreeClassifier (default settings, random_state=20)
3. **K-Nearest Neighbors (KNN)**: StandardScaler + KNeighborsClassifier (default, k=5)
4. **Naive Bayes**: GaussianNB (default settings)
5. **Support Vector Machine (SVM)**: StandardScaler + SVC (RBF kernel, probability=True, random_state=20)

Metrics: Accuracy, AUC (one-vs-rest, weighted), Precision, Recall and F1 (weighted averages over the 7 classes), and MCC.
### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7069 | 0.8827 | 0.7173 | 0.7069 | 0.7101 | 0.6218 |
| Decision Tree | 0.7095 | 0.8060 | 0.7128 | 0.7095 | 0.7105 | 0.6265 |
| KNN | 0.6967 | 0.8751 | 0.7014 | 0.6967 | 0.6939 | 0.6152 |
| Naive Bayes | 0.4859 | 0.7998 | 0.5631 | 0.4859 | 0.4307 | 0.4162 |
| Support Vector Machine | 0.7147 | 0.9071 | 0.7257 | 0.7147 | 0.7163 | 0.6300 |]

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 70.69% and an AUC of 0.8827, showing performance comparable to the other top models.The features were standardised first, so large-valued columns such as area do not dominate the model.It works by creating linear decision boundaries, but may struggle to classify overlapping or complex defect patterns. SVM performed slightly better in AUC (0.9071) and MCC, suggesting that non-linear boundaries may capture the data patterns more effectively. |
| Decision Tree | The Decision Tree achieved an accuracy of 70.95%, which is comparable to the other top models, but its AUC of 0.8060 is lower than Logistic Regression, KNN and SVM. This may be because the fully grown tree produces very confident, mostly 0 or 1 probability scores, resulting in less effective class ranking, even when its predicted labels are correct. It may also overfit the training data, and techniques such as limiting tree depth or pruning could help improve its performance.|
| KNN | KNN achieved an accuracy of 69.67% and an MCC of 0.6152, slightly lower than the best-performing SVM model. It classifies a plate by measuring its distance from the training samples and assigning the majority class among the five nearest neighbours. Since KNN relies on distance, feature standardisation is essential; however, having 27 features and overlapping classes can reduce distance reliability and lead to misclassification, particularly for smaller classes. |
| Naive Bayes | Naive Bayes achieved an accuracy of 48.59% and an MCC of 0.4162, which is considerably lower than the other four models. Its performance may be affected by the assumption that all features are independent, whereas several features in this dataset are strongly correlated, such as area, perimeter and defect dimensions. Additionally, some features may not follow a normal distribution, reducing the accuracy of its probability estimates and leading to incorrect classifications.Its AUC (0.7998) is close to the Decision Tree’s, so it still ranks plates reasonably but often picks the wrong class, probably because correlated features make its probabilities overconfident.|
| Support Vector Machine | SVM achieved the highest scores across all six evaluation metrics, including accuracy (71.47%), AUC (0.9071) and MCC (0.6300). Its accuracy is only marginally higher than Decision Tree ((0.0052 difference, about 2 of the 389 test plates), while the larger AUC gap over Logistic Regression suggests better class separation. The RBF kernel allows SVM to create flexible, non-linear decision boundaries, which may help classify overlapping defect patterns more effectively; however, the small performance difference should be validated through cross-validation before concluding that SVM is significantly better. |
| Overall winner for this dataset | SVM is selected as the best-performing model, as it achieved the highest AUC (0.9071), MCC (0.6300) and accuracy (71.47%) among all five classifiers. However, its recall for Dirtiness (0.45) and Pastry (0.47) is relatively low, indicating that some defects from these classes are still being missed, possibly due to class imbalance. Logistic Regression also offers comparable performance with a simpler model structure, so it remains a reasonable alternative; further cross-validation would help confirm the model selection.Overall accuracy is dominated by the large classes, so it can look good even when small classes are missed, and with only 11 Dirtiness plates in the test set, that recall figure is also noisy.|

## Live app
https://steel-fault-classifier-app.streamlit.app
To try it, download `test_data.csv` from this repository (or use the "Download sample test CSV" button in the app) and upload it.

## Repository contents
- `app.py`: Streamlit web app
- `requirements.txt`: Python dependencies
- `test_data.csv`: test split used by the app (27 features plus the Fault column)
- `model/`: saved trained models (five classifiers, the label encoder and the feature names) as `.joblib` files
- `ML_Assignment_2_Rev_2026aj05730.ipynb`: notebook with data preparation, training and evaluation
- `README.md`: this file]
