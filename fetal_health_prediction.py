# =============================================================================
# PREDICTIVE MODELLING OF FETAL HEALTH USING CTG FEATURES
# Models: Logistic Regression & Support Vector Machine (SVM) with PCA
# =============================================================================

# ── 1. IMPORTING LIBRARIES ───────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler, scale, label_binarize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from itertools import cycle
import scikitplot as skplt


# ── 2. LOADING THE DATASET ───────────────────────────────────────────────────
df = pd.read_csv('fetal_health.csv', header=0)
print(df.shape)


# ── 3. HANDLING MISSING VALUES ───────────────────────────────────────────────
print(df.dropna(inplace=True))


# ── 4. CHECKING FOR OUTLIERS ON CTG FEATURES USING BOX-PLOT ─────────────────
x = df.loc[:, df.columns != 'fetal_health']
sns.boxplot(data=x, orient="v", showmeans=True)
plt.tight_layout()
plt.show()


# ── 5. ADDRESSING OUTLIERS USING Z-SCORES ────────────────────────────────────
z_scores = stats.zscore(df)
df2 = df[(z_scores < 3).all(axis=1)]

# Confirm outliers removed
x = df2.loc[:, df2.columns != 'fetal_health']
sns.boxplot(data=x, orient="v", showmeans=True)
plt.tight_layout()
plt.show()


# ── 6. DROPPING UNWANTED CTG FEATURES ────────────────────────────────────────
desired_var = df2.drop(['fetal_movement', 'uterine_contractions', 'severe_decelerations'],
                       axis=1)
print(desired_var)


# ── 7. CORRELATION HEATMAP ────────────────────────────────────────────────────
corrmat = desired_var.corr()
print(corrmat)

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(corrmat, cmap='coolwarm', annot=True)
plt.tight_layout()
plt.show()


# ── 8. DESCRIPTIVE SUMMARY ────────────────────────────────────────────────────
print(desired_var.describe())


# ── 9. FETAL HEALTH OUTCOME DISTRIBUTION ─────────────────────────────────────
sns.countplot(x="fetal_health", data=desired_var)
plt.title("Fetal Health Outcome Distribution")
plt.show()


# =============================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================

# ── 10. SEPARATING FEATURES AND LABEL ────────────────────────────────────────
feature = desired_var.loc[:, desired_var.columns != 'fetal_health']
label = desired_var['fetal_health']


# ── 11. STANDARDISING THE CTG FEATURES ───────────────────────────────────────
object = scale(feature)
object = pd.DataFrame(object, index=feature.index, columns=feature.columns)
print(object.apply(np.mean))
print(object.apply(np.std))


# ── 12. FITTING THE PCA MODEL ─────────────────────────────────────────────────
pca = PCA().fit(object)
print(pca)

# Kaiser's criterion – retain components with eigenvalue > 1
X_r = pca.fit(object).transform(object)
print('\nEigenvalues \n%s' % pca.explained_variance_)
print('Eigenvectors \n%s' % pca.components_)


# ── 13. SELECTING PRINCIPAL COMPONENTS ───────────────────────────────────────

# Method 1 – Scree Plot
def screeplot(pca, object_values):
    y = np.std(pca.transform(object_values), axis=0) ** 2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "*-")
    plt.xticks(x, ["Principal component " + str(i) for i in x], rotation=60)
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.show()

screeplot(pca, object)


# Method 2 – Variance Information (Cumulative Explained Variance)
def var_explained():
    import numpy as np
    from matplotlib.pyplot import figure, show
    from matplotlib.ticker import MaxNLocator

    ax = figure().gca()
    ax.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axvline(x=6, linewidth=1, color='r', alpha=0.5)
    plt.title('Explained Variance of PCA by Component')
    show()

var_explained()


# ── 14. PCA VISUALISATION ─────────────────────────────────────────────────────
target_names = label

pca = PCA(n_components=4)
pca_model = pca.fit(object).transform(object)

plt.figure()
colors = ['navy', 'turquoise', 'red']
lw = 2

for color, i, target_name in zip(colors, [1, 2, 3], target_names):
    plt.scatter(pca_model[label == i, 0], pca_model[label == i, 1],
                color=color, alpha=0.8, lw=lw, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of CTG-FEATURE dataset')
plt.show()


# =============================================================================
# MODEL 1 – LOGISTIC REGRESSION WITH PCA
# =============================================================================

# ── 15. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
feature = desired_var.loc[:, desired_var.columns != 'fetal_health']
label = desired_var['fetal_health']

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2, random_state=42)


# ── 16. APPLYING PCA ON TRAINING DATA ────────────────────────────────────────
pca = PCA(n_components=6)
PCA_comp = pca.fit_transform(feature_train)
principalDF = pd.DataFrame(data=PCA_comp,
                            columns=['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6'])


# ── 17. FITTING LOGISTIC REGRESSION ──────────────────────────────────────────
logistic_clf = LogisticRegression()
logistic_clf.fit(principalDF, label_train)


# ── 18. PREDICTING ON TEST DATA ───────────────────────────────────────────────
pca = PCA(n_components=6)
PCA_comp = pca.fit_transform(feature_test)
pcaTESTDF = pd.DataFrame(data=PCA_comp,
                          columns=['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6'])

prediction = logistic_clf.predict(pcaTESTDF)
print(prediction)


# ── 19. LOGISTIC REGRESSION – MODEL EVALUATION ───────────────────────────────
print("Accuracy:", accuracy_score(label_test, prediction))
print(pd.DataFrame(confusion_matrix(label_test, prediction)))
print(classification_report(label_test, prediction))


# ── 20. LOGISTIC REGRESSION – ROC-AUC CURVE ──────────────────────────────────
# Binarize the output
y_test_bin = label_binarize(label_test, classes=[1, 2, 3])

# Predict probabilities on the test set
prediction = logistic_clf.predict_proba(pcaTESTDF)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(logistic_clf.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(len(logistic_clf.classes_)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve for Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# =============================================================================
# MODEL 2 – SUPPORT VECTOR MACHINE (SVM) WITH PCA
# =============================================================================

# ── 21. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
feature = desired_var.loc[:, desired_var.columns != 'fetal_health']
label = desired_var['fetal_health']

feature_train, feature_test, label_train, label_test = train_test_split(
    feature, label, test_size=0.2, random_state=42)


# ── 22. APPLYING PCA ON TRAINING DATA ────────────────────────────────────────
pca = PCA(n_components=6)
PCA_comp = pca.fit_transform(feature_train)
principalDF2 = pd.DataFrame(data=PCA_comp,
                             columns=['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6'])


# ── 23. FITTING SVM ACROSS ALL KERNELS ───────────────────────────────────────
# Apply PCA on test data
pca = PCA(n_components=6)
PCA_comp = pca.fit_transform(feature_test)
pcaTESTDF2 = pd.DataFrame(data=PCA_comp,
                           columns=['PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6'])

kernel = ['linear', 'rbf', 'sigmoid', 'poly']
for i in kernel:
    model = SVC(kernel=i, C=0.5, probability=True)
    model.fit(principalDF2, label_train)
    print("for kernel:", i)
    print('Accuracy is:', model.score(pcaTESTDF2, label_test))


# ── 24. PREDICTING ON TEST DATA ───────────────────────────────────────────────
prediction = model.predict(pcaTESTDF2)
print(prediction)


# ── 25. SVM – MODEL EVALUATION ───────────────────────────────────────────────
print("Accuracy:", accuracy_score(label_test, prediction))
print(pd.DataFrame(confusion_matrix(label_test, prediction)))
print(classification_report(label_test, prediction))


# ── 26. SVM – ROC-AUC CURVE ──────────────────────────────────────────────────
# Binarize the output
y_test_bin = label_binarize(label_test, classes=[1, 2, 3])

# Predict probabilities on the test set
prediction = model.predict_proba(pcaTESTDF2)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(model.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(len(model.classes_)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve for SVM Classifier')
plt.legend(loc="lower right")
plt.show()


# =============================================================================
# HYPERTUNING SVM USING GRIDSEARCH
# =============================================================================

# ── 27. IMPORTING LIBRARIES ───────────────────────────────────────────────────
# (already imported above: GridSearchCV, SVC)

# ── 28. DEFINE PARAMETER GRID ─────────────────────────────────────────────────
param_grid = {
    'C': [0.1, 0.5, 1, 4, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4],
}

# ── 29. CREATE SVM CLASSIFIER ─────────────────────────────────────────────────
svm = SVC()
print(svm)

# ── 30. PERFORM GRID SEARCH WITH CROSS-VALIDATION ────────────────────────────
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid,
                           cv=5, scoring='accuracy')
grid_search.fit(feature_train, label_train)

# ── 31. GET BEST PARAMETERS ───────────────────────────────────────────────────
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
