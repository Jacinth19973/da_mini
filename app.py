from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and prepare the dataset
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')  # Load the Excel file
    features = ['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'uv_index', 'visibility_km']
    target = 'condition_text'
    
    df = df[features + [target]]  # Keep only necessary columns
    df.dropna(inplace=True)  # Drop rows with NaN

    # Encode categorical target variable (weather condition)
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    
    return df, le

# Train the CART Decision Tree model
def train_cart_model(file_path):
    df, le = load_and_prepare_data(file_path)
    X = df[['temperature_celsius', 'wind_kph', 'humidity', 'pressure_mb', 'uv_index', 'visibility_km']]
    y = df['condition_text']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    cart_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_split=10, min_samples_leaf=5)
    cart_tree.fit(X_train, y_train)

    # Save the model and label encoder
    joblib.dump(cart_tree, 'cart_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

# Load the trained CART model
def load_cart_model():
    cart_model = joblib.load('cart_model.pkl')
    le = joblib.load('label_encoder.pkl')
    return cart_model, le

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the user uploaded a file
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file
        file_path = 'weather_dataset.xlsx'  # Save the uploaded file as this name
        file.save(file_path)

        # Train the model with the uploaded dataset
        train_cart_model(file_path)

        return "Model trained successfully with the uploaded dataset!"

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    temperature = float(request.form['temperature'])
    wind_kph = float(request.form['wind_kph'])
    humidity = float(request.form['humidity'])
    pressure_mb = float(request.form['pressure_mb'])
    uv_index = float(request.form['uv_index'])
    visibility_km = float(request.form['visibility_km'])

    # Prepare input data for the model
    features = pd.DataFrame({
        'temperature_celsius': [temperature],
        'wind_kph': [wind_kph],
        'humidity': [humidity],
        'pressure_mb': [pressure_mb],
        'uv_index': [uv_index],
        'visibility_km': [visibility_km],
    })

    # Load the model and make a prediction
    cart_model, le = load_cart_model()
    prediction = cart_model.predict(features)
    predicted_condition = le.inverse_transform(prediction)

    return render_template('result.html', prediction=predicted_condition[0])

if __name__ == '__main__':
    app.run(debug=True)