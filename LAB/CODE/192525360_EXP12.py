import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
ax[0].set_xlabel("Sepal Length (cm)")
ax[0].set_ylabel("Sepal Width (cm)")
ax[0].set_title("Iris Flower Classification using KNN")
ax[0].text(0.5, -0.15, "Setosa     Versicolor     Virginica", transform=ax[0].transAxes, ha='center')
ax[1].axis('off')
output = "IRIS FLOWER CLASSIFICATION USING KNN\n\n"
output += "Accuracy: " + str(round(accuracy * 100, 2)) + "%\n\n"
output += "Classification Report:\n" + report
ax[1].text(0, 1, output, fontsize=10, family='monospace', verticalalignment='top')
ax[1].set_title("Console Output")
plt.tight_layout()
plt.show()
