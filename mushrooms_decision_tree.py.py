# mushrooms_decision_tree.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    """Muat dataset dan lakukan label encoding untuk semua kolom kategori."""
    df = pd.read_csv(csv_path)
    le_dict = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data, latih Decision Tree, dan kembalikan model + metrik evaluasi."""
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["edible", "poisonous"])
    return clf, acc, report, X.columns

def visualize_tree(model: DecisionTreeClassifier, feature_names, class_names=["edible", "poisonous"]):
    """Visualisasikan Decision Tree menggunakan matplotlib."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Decision Tree Visualization for Mushroom Classification")
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load & preprocess
    df, encoders = load_and_preprocess("mushrooms.csv")
    print("✅ Data loaded and preprocessed.")

    # 2. Train & evaluate
    clf, accuracy, report, features = train_and_evaluate(df, test_size=0.2)
    print(f"\n📊 Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # 3. Visualize
    visualize_tree(clf, features)

if __name__ == "__main__":
    main()
