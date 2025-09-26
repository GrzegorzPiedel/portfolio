# Housing Prices Dashboard – USA

## Project Description
An interactive dashboard for exploring and predicting housing prices in the USA. The project includes:

- Analysis of housing data (area, number of rooms, bathrooms, floors, furnishing, etc.)
- Visualizations: histograms, scatter plots, and box plots
- An interactive linear regression model to predict house prices based on selected features

Data source: Kaggle [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/code).

---

## Technologies
- Python 3.x  
- Streamlit  
- Pandas  
- Plotly  
- scikit-learn  
- NumPy  

---

## Installation

1. Clone the repository:
```bash
git clone <YOUR_REPO_URL>
cd <REPO_FOLDER>```

2. Create a virtual environment:
python -m venv venv

3. Activate the virtual environment:
- Windows:
venv\Scripts\activate

- Mac/Linux:
source venv/bin/activate

4. Install dependencies:
pip install -r requirements.txt

## Running the App
Start the Streamlit application:
streamlit run app.py

The dashboard will open in your browser at http://localhost:8501.

Notes

To change the CSV file path, make sure app.py points to the correct location relative to the repository:

df = pd.read_csv('data/Housing.csv')


Trendline in scatter plots requires the statsmodels library:

pip install statsmodels

Author

Grzegorz Piedel
