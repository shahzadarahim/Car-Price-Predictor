import pickle
from flask import Flask, jsonify, url_for, request, app, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

app = Flask(__name__) # create an instance
model = pickle.load(open('./models/Model.pkl', 'rb'))
Brand_Encoder = pickle.load(open('./models/Brand_Encoder.pkl', 'rb'))
Model_Encoder = pickle.load(open('./models/Model_Encoder.pkl', 'rb'))
OneHot_Encoder = pickle.load(open('./models/OneHot_Encoder.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Brand = request.form['Brand']
        Model = request.form['Model']
        Fuel = request.form['Fuel_Type']
        Transmission = request.form['Transmission']
        Year = int(request.form['Year'])
        EngineSize = float(request.form['EngineSize'])
        Mileage = int(request.form['Mileage'])
        Doors = int(request.form['Doors'])
        OwnerCount = int(request.form['OwnerCount'])

        # Create a DataFrame for the input data
        input_df = pd.DataFrame({
            'Brand' : [Brand],
            'Model' : [Model],
            'Fuel_Type' : [Fuel],
            'Transmission' : [Transmission],
            'Year' : [Year],
            'EngineSize' : [EngineSize],
            'Mileage' : [Mileage],
            'Doors' : [Doors],
            'OwnerCount' : [OwnerCount]
        })

        # Apply mean encoding for Brand and Model
        input_df['Brand'] = input_df['Brand'].map(Brand_Encoder)
        input_df['Model'] = input_df['Model'].map(Model_Encoder)
        input_df['Brand'] = input_df['Brand'].fillna(input_df['Brand'].mean())
        input_df['Model'] = input_df['Model'].fillna(input_df['Model'].mean())
        # input_df.drop(['Brand', 'Model'], axis=1, inplace=True)

        # One Hot encode Fuel and Transmission
        # input_df.rename(columns={'Fuel': 'Fuel_Type'}, inplace=True)
        cat_columns = ['Transmission', 'Fuel_Type']
        encoded_array = OneHot_Encoder.transform(input_df[cat_columns])
        encoded_df = pd.DataFrame(encoded_array, 
                                  columns = OneHot_Encoder.get_feature_names_out(cat_columns),
                                  index = input_df.index)
        
        # Merge encoded columns with input data
        input_df_encoded = input_df.drop(columns=cat_columns)
        input_data = pd.concat([input_df_encoded, encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', prediction_texts='Sorry you cannot sell this car')
        else:
            return render_template('index.html', prediction_texts='Car is worth at: $ {}'.format(output))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)