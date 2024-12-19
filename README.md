# King_County_House_Price_Analysis


## Project Overview
This project involves analyzing and predicting housing prices in King County, USA, using data from house sales between May 2014 and May 2015. The dataset includes features such as square footage, number of bedrooms, number of floors, and other relevant attributes. The goal of this analysis is to determine the market price of a house given a set of features and to build predictive models to estimate housing prices.

The project demonstrates proficiency in data wrangling, exploratory data analysis, model development, and model evaluation using Python libraries such as Pandas, Matplotlib, Seaborn, Scikit-learn, and others.

---

## Dataset Description
The dataset contains the following features:

| Variable         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `id`             | A unique identifier for a house                                            |
| `date`           | Date the house was sold                                                    |
| `price`          | Price of the house (target variable)                                       |
| `bedrooms`       | Number of bedrooms                                                        |
| `bathrooms`      | Number of bathrooms                                                       |
| `sqft_living`    | Square footage of the home                                                |
| `sqft_lot`       | Square footage of the lot                                                 |
| `floors`         | Total floors in the house                                                 |
| `waterfront`     | Whether the house has a waterfront view                                   |
| `view`           | Number of times the house has been viewed                                 |
| `condition`      | Overall condition of the house                                            |
| `grade`          | Grade of the house based on King County grading system                   |
| `sqft_above`     | Square footage of the house excluding the basement                        |
| `sqft_basement`  | Square footage of the basement                                            |
| `yr_built`       | Year the house was built                                                  |
| `yr_renovated`   | Year the house was renovated                                              |
| `zipcode`        | Zip code of the house location                                            |
| `lat`            | Latitude coordinate of the house                                          |
| `long`           | Longitude coordinate of the house                                         |
| `sqft_living15`  | Square footage of living area in 2015 (post-renovation)                   |
| `sqft_lot15`     | Lot size area in 2015 (post-renovation)                                   |

---

## Project Steps

### 1. Data Wrangling
- **Removing unnecessary columns**: Dropped the `id` and `Unnamed: 0` columns as they do not contribute to the analysis.
- **Handling missing values**: Replaced missing values in `bedrooms` and `bathrooms` columns with their respective means.

```python
# Drop unnecessary columns
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

# Replace missing values
df['bedrooms'].fillna(df['bedrooms'].mean(), inplace=True)
df['bathrooms'].fillna(df['bathrooms'].mean(), inplace=True)
```

### 2. Exploratory Data Analysis (EDA)

#### Unique Floor Values
Used the `value_counts()` method to count the number of houses with unique floor values.
```python
floor_counts = df['floors'].value_counts().to_frame()
```

#### Boxplot of Waterfront View vs. Price
Explored the relationship between waterfront views and house prices using a boxplot.
```python
sns.boxplot(x='waterfront', y='price', data=df)
plt.title("Waterfront View vs. Price")
plt.show()
```

#### Correlation Between Features
Determined the correlation between `sqft_above` and `price` using a regression plot.
```python
sns.regplot(x='sqft_above', y='price', data=df)
plt.title("Correlation Between Sqft Above and Price")
plt.show()
```
Identified the feature most correlated with `price`:
```python
correlations = df.corr()['price'].sort_values(ascending=False)
```

### 3. Model Development

#### Linear Regression

**Model 1**: Predict `price` using the `long` feature.
```python
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X, Y)
r2_long = lm.score(X, Y)
```

**Model 2**: Predict `price` using `sqft_living`.
```python
X = df[['sqft_living']]
Y = df['price']
lm.fit(X, Y)
r2_sqft_living = lm.score(X, Y)
```

**Model 3**: Predict `price` using multiple features.
```python
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]
Y = df['price']
lm.fit(X, Y)
r2_multiple = lm.score(X, Y)
```

#### Polynomial Regression Pipeline
Created a pipeline for polynomial regression with scaling.
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

Input = [('scale', StandardScaler()),
         ('polynomial', PolynomialFeatures(include_bias=False)),
         ('model', LinearRegression())]

pipe = Pipeline(Input)
pipe.fit(X, Y)
r2_pipeline = pipe.score(X, Y)
```

### 4. Model Evaluation and Refinement

#### Ridge Regression
Used Ridge regression with a regularization parameter of 0.1.
```python
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
r2_ridge = ridge_model.score(x_test, y_test)
```

#### Polynomial Ridge Regression
Performed a second-order polynomial transform and applied Ridge regression.
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
ridge_poly_model = Ridge(alpha=0.1)
ridge_poly_model.fit(x_train_poly, y_train)
r2_poly_ridge = ridge_poly_model.score(x_test_poly, y_test)
```



## Results

- **Most correlated feature with price**: `sqft_living`.
- **R² values**:
  - Linear Regression (Longitude): {r2_long}
  - Linear Regression (Sqft Living): {r2_sqft_living}
  - Multiple Linear Regression: {r2_multiple}
  - Ridge Regression: {r2_ridge}
  - Polynomial Ridge Regression: {r2_poly_ridge}



## Conclusion
This project demonstrates an end-to-end workflow for analyzing and predicting housing prices. By leveraging data wrangling, exploratory data analysis, and machine learning techniques, we were able to build predictive models and derive actionable insights about the King County housing market.

---

## Tools and Libraries
- **Pandas**: Data manipulation and analysis.
- **Matplotlib/Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning and model evaluation.
- **Jupyter Notebook**: Interactive coding environment.

---

## Repository Structure
```
king_county_housing_analysis/
├── data/
│   └── kc_house_data_NaN.csv
├── notebooks/
│   └── house_sales_analysis.ipynb
├── README.md
└── results/
    ├── correlation_matrix.png
    └── regression_plots.png



