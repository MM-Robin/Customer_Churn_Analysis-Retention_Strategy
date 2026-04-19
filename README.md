# Customer Churn Analysis & Retention Strategy

## Overview

This project performs comprehensive analysis on customer churn data from a telecom company to identify key factors influencing churn and develop effective retention strategies. The analysis includes data exploration, predictive modeling, and visualization to provide actionable insights for business decision-making.

## Features

- **Data Analysis**: Exploratory data analysis with statistical insights and visualizations
- **Churn Prediction**: Machine learning model to predict customer churn probability
- **Interactive Dashboard**: Web-based dashboard for exploring churn patterns
- **Retention Strategy Recommendations**: Data-driven suggestions for customer retention

## Project Structure

```
├── .gitignore               # Git ignore rules
├── README.md               # This file
├── requirements.txt         # Python dependencies
├── analysis.py              # Main analysis and modeling script
├── dashboard.py             # Interactive dashboard
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Customer churn dataset
├── notebooks/               # Jupyter notebooks for exploration
└── output/                  # Generated outputs (charts, reports, predictions)
    ├── *.png                # Visualization charts
    ├── churn_predictions.csv # Model predictions
    ├── churn_business_report.xlsx # Excel report
    └── churn_presentation.pptx   # PowerPoint presentation
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd Customer_Churn_Analysis&Retention_Strategy
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Analysis

```bash
python analysis.py
```

This will perform data preprocessing, build the churn prediction model, and generate visualizations in the `output/` directory.

### Launching the Dashboard

```bash
python dashboard.py
```

Opens an interactive web dashboard for exploring churn insights.

business reports and presentations in the `output/` directory.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- streamlit (for dashboard)

## Dataset

The analysis uses the Telco Customer Churn dataset from IBM Watson Analytics, which includes:

- Customer demographics
- Service usage patterns
- Billing information
- Churn status

## Results

The project generates:

- Churn prediction model with performance metrics
- Key factor analysis for churn drivers
- Retention strategy recommendations
- Interactive visualizations and reports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Author

**Mainuddin Monsur Robin**
