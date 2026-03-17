import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()

model.fit(X, y)

pickle.dump(model, open("crop_model.pkl", "wb"))

print("Model trained and saved")
