from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

app = Flask(__name__)

def load_model(model_filename: str):
    try:
        model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_dataset(dataset_filename: str, delimiter: str = ';'):
    try:
        dataset = pd.read_csv(dataset_filename, delimiter=delimiter)
        print(f"Dataset loaded from {dataset_filename}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def predictor(nama_makanan: str, dataset: pd.DataFrame, scaler: StandardScaler, model, target_scaler: MinMaxScaler, poly: PolynomialFeatures, portion_size: float) -> pd.DataFrame:
    columns_to_convert = ['Air', 'PROTEIN', 'LEMAK', 'KH', 'SERAT', 'ABU', 'KALSIUM', 'FOSFOR',
                          'BESI', 'NATRIUM', 'KALIUM', 'TEMBAGA', 'SENG', 'RETINOL', 'B-KAR',
                          'KAR -TOTAL', 'THIAMIN', 'RIBOFLAVIN', 'NIASIN', 'VIT_C', 'BDD']
    
    nama_makanan = nama_makanan.lower()
    searched_data = dataset[dataset["Nama_bahan"] == nama_makanan]
    if searched_data.empty:
        raise ValueError(f"No data found for the food item: {nama_makanan}")

    numeric_value = searched_data[columns_to_convert].to_numpy()
    numeric_value_df = pd.DataFrame(numeric_value, columns=columns_to_convert)  # Buat DataFrame dengan nama kolom yang benar
    numeric_value_poly = poly.transform(numeric_value_df)
    numeric_value_poly_df = pd.DataFrame(numeric_value_poly, columns=poly.get_feature_names_out(columns_to_convert))  # Tambahkan ini
    numeric_value_scaled = scaler.transform(numeric_value_poly_df)  # Ubah ini
    energi_scaled = model.predict(numeric_value_scaled)
    energi = target_scaler.inverse_transform(energi_scaled.reshape(-1, 1))

    predicted_nutrition = pd.DataFrame({
        'ENERGI': np.round(energi.flatten() * (portion_size / 100), 2),
        'PROTEIN': np.round(searched_data['PROTEIN'].values * (portion_size / 100), 2),
        'LEMAK': np.round(searched_data['LEMAK'].values * (portion_size / 100), 2)
    })

    return predicted_nutrition

# Load model and dataset
model_filename = "best_rf_model.pkl"
dataset_filename = "Dataset_Giziwise.csv"

model = load_model(model_filename)
dataset = load_dataset(dataset_filename)

if model is not None and dataset is not None:
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    scaler = StandardScaler()
    target_scaler = MinMaxScaler()

    columns_to_convert = ['Air', 'PROTEIN', 'LEMAK', 'KH', 'SERAT', 'ABU', 'KALSIUM', 'FOSFOR',
                          'BESI', 'NATRIUM', 'KALIUM', 'TEMBAGA', 'SENG', 'RETINOL', 'B-KAR',
                          'KAR -TOTAL', 'THIAMIN', 'RIBOFLAVIN', 'NIASIN', 'VIT_C', 'BDD']

    dataset['Nama_bahan'] = dataset['Nama_bahan'].str.lower()
    dataset[columns_to_convert] = dataset[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    dataset_cleaned = dataset.dropna()

    features_poly = poly.fit_transform(dataset_cleaned[columns_to_convert])
    scaler.fit(features_poly)
    target_scaler.fit(dataset_cleaned['ENERGI'].values.reshape(-1, 1))
else:
    raise Exception("Failed to load model or dataset.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    nama_makanan = data.get('nama_makanan')
    portion_size = float(data.get('portion_size'))

    try:
        result = predictor(nama_makanan, dataset_cleaned, scaler, model, target_scaler, poly, portion_size)
        response = {
            #'nama_makanan': nama_makanan,
            'ENERGI': result['ENERGI'].values[0],
            'PROTEIN': result['PROTEIN'].values[0],
            'LEMAK': result['LEMAK'].values[0]
        }
        return jsonify(response)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
