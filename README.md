# Crop Recommendation System

This is a machine learning-based web application that recommends the most suitable crops to grow based on various environmental parameters. The system uses a Random Forest Classifier to make predictions based on soil and climate conditions.

## Features

- Web-based interface for easy interaction
- Real-time crop recommendations
- Uses machine learning to predict suitable crops
- Takes into account multiple parameters:
  - Nitrogen content in soil
  - Phosphorus content in soil
  - Potassium content in soil
  - Temperature
  - Humidity
  - pH level
  - Rainfall

## Tech Stack

- Python 3.x
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas (Data Processing)
- NumPy (Numerical Operations)
- HTML/CSS (Frontend)

## Project Structure

```
Crop_Recommendation-main/
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── model.pkl             # Trained model
├── minmaxscaler.pkl      # Min-Max scaler for data preprocessing
├── standscaler.pkl       # Standard scaler for data preprocessing
├── retrain_model.py      # Script for retraining the model
├── Crop_recommendation.csv    # Dataset
├── templates/            # HTML templates
├── static/              # Static files (CSS, JS, images)
└── .venv/               # Virtual environment
```

## Installation & Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Crop_Recommendation-main
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## How to Use

1. Access the web interface through your browser
2. Enter the following parameters:
   - Nitrogen content in soil
   - Phosphorus content in soil
   - Potassium content in soil
   - Temperature (in Celsius)
   - Humidity (in %)
   - pH level
   - Rainfall (in mm)
3. Click on the predict button
4. Get the crop recommendation based on your input parameters

## Model Information

The system uses a Random Forest Classifier trained on a dataset containing various crop parameters. The model:
- Uses both MinMax and Standard scaling for feature normalization
- Is trained with 100 decision trees
- Provides predictions for multiple crop types based on the input parameters

## Dataset

The system uses the `Crop_recommendation.csv` dataset which contains:
- Soil parameters (N, P, K values)
- Climate parameters (temperature, humidity, rainfall)
- pH values
- Corresponding crop labels

## Retraining the Model

If you want to retrain the model with new data:
1. Update the `Crop_recommendation.csv` file with new data
2. Run:
```bash
python retrain_model.py
```

## Requirements

Key dependencies include:
- Flask==3.1.0
- numpy==2.2.4
- pandas==2.2.3
- scikit-learn==1.6.1
- And other packages listed in requirements.txt

## Development

The application is built with Flask and follows a simple MVC pattern:
- Model: Random Forest Classifier
- View: HTML templates in the templates directory
- Controller: Flask routes in app.py

## Contributing

Feel free to fork the repository and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 