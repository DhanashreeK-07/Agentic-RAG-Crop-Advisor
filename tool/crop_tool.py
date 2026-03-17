import pickle
import numpy as np

model = pickle.load(open("crop_model.pkl","rb"))

def crop_prediction_tool(data):

    arr = np.array(data).reshape(1,-1)

    crop = model.predict(arr)[0]

    return f"Predicted crop: {crop}"
