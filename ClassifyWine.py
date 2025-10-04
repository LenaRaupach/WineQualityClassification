from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def get_training_data():
    try:
        return load_wine()
    except:
        print("Failed to load wine training data.")
        return {}

def get_decision_tree_classifier(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def visualize_tree(wine, clf):
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=wine.feature_names[:5], class_names=["schlecht", "gut"], filled=True)
    plt.show()

def get_metrics():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Genauigkeit (Accuracy): {accuracy:.2f}")
    print(f"Präzision (Precision): {precision:.2f}")
    print(f"Recall (Trefferquote): {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC-ROC: {auc:.2f}")

if __name__ == '__main__':
    wine = get_training_data()

    # use only the first five properties 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium'
    X = wine.data[:, :5] # feature values
    y = (wine.target == 0).astype(int)  # result for test data. The class 0  represents result "good", the class 1 represents any other result

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = get_decision_tree_classifier(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    visualize_tree(wine, clf)
    get_metrics()

