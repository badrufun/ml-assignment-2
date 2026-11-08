# ML Assignment 2 - Steel Plate Fault Classification

## Problem statement

I trained five classifiers to identify the type of defect on a steel plate. Each row describes a defect that has already been found. The goal is to classify that defect from measurements such as area, perimeter, and brightness.

## Dataset

I used the [Steel Plates Faults dataset from UCI](https://archive.ics.uci.edu/dataset/198/steel%2Bplates%2Bfaults). It has 1,941 records, 27 numeric input features, and seven fault types. There are no missing values in the source data. The classes are uneven. Other Faults has 673 records, while Dirtiness has 55.

The source has seven indicator columns, one for each fault type. I combined them into one target column called `Fault` and excluded all seven indicators from the inputs. I used a stratified 80/20 split with random seed 42, giving 1,552 training rows and 389 test rows. The same split was used for every model.

## GitHub repository

[github.com/badrufun/ml-assignment-2](https://github.com/badrufun/ml-assignment-2)

## Models and results

I used StandardScaler for Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, and the RBF Support Vector Machine. The scaler was fitted only on the training rows and saved with each model. The Decision Tree used the original feature values.

I report macro precision, recall, and F1, which give each fault type the same weight. AUC is the macro average of the one-vs-rest ROC AUC scores calculated from predicted probabilities. MCC is calculated across all seven classes.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7455 | 0.8580 | 0.7461 | 0.7642 | 0.7523 | 0.6733 |
| K-Nearest Neighbors | 0.7275 | 0.9184 | 0.7458 | 0.7482 | 0.7416 | 0.6548 |
| Gaussian Naive Bayes | 0.6093 | 0.9225 | 0.6333 | 0.7169 | 0.6339 | 0.5484 |
| Support Vector Machine | 0.7506 | 0.9493 | 0.8006 | 0.7390 | 0.7626 | 0.6765 |

- **Logistic Regression** had the second-highest AUC. Its accuracy tied with K-Nearest Neighbors, and its macro precision was slightly higher.
- **Decision Tree** had the highest macro recall and came close to the SVM on accuracy and MCC. It had the lowest AUC.
- **K-Nearest Neighbors** had higher macro recall than Logistic Regression, with almost the same macro F1.
- **Gaussian Naive Bayes** had an AUC of 0.9225 but the lowest accuracy and MCC. It ranked the classes better than its final predictions suggest.
- **Support Vector Machine** had the highest accuracy, AUC, macro precision, macro F1, and MCC.

I chose the SVM as the overall winner because it had the best macro F1 and MCC. The Decision Tree was close and had better macro recall. This comparison is based on one held-out split.

## Live app

[Steel Plate Fault Classification on Streamlit](https://cruouklzcc5di4pphwrphx.streamlit.app/)

## Run the app locally

I used Python 3.12. From the project directory, run the following commands in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` for the Python commands.

Open http://localhost:8501 and select a model. The app uses the included `test_data.csv` by default. Upload a different labeled CSV to evaluate it instead. The app shows the six evaluation metrics, a confusion matrix, and a classification report with precision, recall, F1, and support for each fault type. It loads the saved model from `model/` and uses the `Fault` column as the true label.

To reproduce the trained models and test CSV, run the training script. It downloads the source data from UCI.

```powershell
.\.venv\Scripts\python.exe train.py
```

## Dataset credit

Buscema, M., Terzi, S., and Tastle, W. (2010). Steel Plates Faults. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5J88N). The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The included CSV contains the held-out rows and replaces the seven fault indicators with the single `Fault` label.
