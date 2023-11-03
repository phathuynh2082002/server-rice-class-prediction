# Khai báo thư viện
import pickle
import pandas as pd
import numpy as np
from rice import standar_data_predict
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

# Khởi tạo Flask Sever Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] =  'Content-Type'


# Khởi tạo đường dẫn dự đoán
@app.route('/predict', methods=['post'])
@cross_origin(origins='*')
def predict_modal():
    form = request.form.to_dict(flat=False)
    data = pd.DataFrame.from_dict(form)
    data = data.astype(float)
    loaded_model = pickle.load(open('sup_vector_model.sav', 'rb'))
    data_predict = standar_data_predict(data)
    print(data_predict['Area'])
    result = loaded_model.predict(data_predict)
    return result[0]

@app.route('/', methods=['get'])
@cross_origin(origins='*')
def abc():
    return '1'

if __name__ == '__main__':
    app.debug = True
    app.run(port='1000')
