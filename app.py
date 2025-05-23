import pickle
from flask import Flask, request, render_template
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Getting form inputs
            Brand = request.form['Brand']
            Model = request.form['Model']
            Fuel = request.form['Fuel_Type']
            Transmission = request.form['Transmission']
            Year = int(request.form['Year'])
            Engine_Size = float(request.form['Engine_Size'])
            Mileage = int(request.form['Mileage'])
            Doors = int(request.form['Doors'])
            Owner_Count = int(request.form['Owner_Count'])

            # Create a DataFrame for the input data
            input_df = pd.DataFrame({
                'Brand' : [Brand],
                'Model' : [Model],
                'Fuel_Type' : [Fuel],
                'Transmission' : [Transmission],
                'Year' : [Year],
                'Engine_Size' : [Engine_Size],
                'Mileage' : [Mileage],
                'Doors' : [Doors],
                'Owner_Count' : [Owner_Count]
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
            output = 80*(round(prediction[0]))

            if output < 0:
                return render_template('index.html', prediction_text='Sorry you cannot sell this car')
            else:
                return render_template('index.html', prediction_text='Car is worth at: ₹ {}'.format(output))
        except KeyError as e:
            error_message = f"Error with column: {e}. Please make sure all fields are filled correctly."
            return render_template('index.html', prediction_text=error_message)
        
        except ValueError as e:
            error_message = f"Invalid data: {e}. Please check the input values."
            return render_template('index.html', prediction_text=error_message)
        
        except Exception as e:
            error_message = f"An unexpected error occured: {str(e)}"
            return render_template('index.html', prediction_text=error_message)

    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)