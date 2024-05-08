# Advanced Forecasting for Airbnb Competitive Rental Pricing

## Introduction
The project focuses on developing a predictive model for rental prices in various cities, using data such as property type, room type, and other relevant features. This endeavor was chosen due to its potential to offer valuable insights into the rental market, enabling both renters and landlords to make informed decisions. 

For renters, it demystifies the rental pricing landscape, aiding them in finding accommodations that fit their budget and preferences. Landlords, on the other hand, can optimize their pricing strategies to remain competitive while ensuring profitability. Furthermore, this model can contribute to a more transparent and efficient rental market, potentially influencing policy-making and urban planning by providing data-driven insights into housing trends. 

## Dataset
The dataset contains 74,111 listings, each with 29 attributes detailing the property, host, and booking policies, focused on predicting the logarithm of the renting price.
It can be retrieved from Kaggle at: [Kaggle Dataset](https://www.kaggle.com/datasets/rupindersinghrana/airbnb-price-dataset/data).

## Method

### Data Observation

1. The boxplot reveals the presence of outliers, makes the median a more reliable measure than the mean for imputing missing values.
<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/DataExploration1.png" alt="Figure 1" width="50%">
</div>

2. The histogram of the distribution of Airbnb price shows the data is a little left-skewed, so we might choose some machine learning models are more robust to skewness like Random Forests or Gradient Boosting Machines.
<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/DataExploration3.png" alt="Figure 2" width="50%">
</div>

3. The correlation matrix and pairplot is calculated below.
<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/DataExploration4.png" alt="Figure 3" width="45%">
  <img src="graphs/DataExploration5.jpg" alt="Figure 3" width="45%">
</div>

***More investigations of other attributes can be found in the [Data PreProcessing Notebook](Data_preprocessing.ipynb), which are not shown due to the limit of space.***

### Data Preprocessing

1. Data Cleaning: 
   1. We drop columns with excessive unique categories or missing data that would be impractical to one-hot encode or impute, such as 'thumbnail_url', 'zipcode', 'neighbourhood', 'first_review', and 'last_review'.
   2. We convert the id column to a numeric type and rename it to 'id' for clarity and ease of reference and extraction.


2. Handling Missing Values: 
   1. Based on the presence of outliers, median values are chosen as a more reliable measure for imputing missing values. Missing values in 'bathrooms', 'bedrooms', and 'beds' are imputed with the median of each column, grouped by the 'accommodates' category, to maintain the integrity of the data.
   2. Missing values in 'host_response_rate' and 'review_scores_rating' are imputed with their respective median values. Before imputation, the 'host_response_rate' is converted from a percentage string to a float.
   3. We drop latitude and longitude column because it is difficult to process this kind of data and we already have City column.

3. Bag of Words (BOW) & Term Frequency-Inverse Document Frequency(TF-IDF)
   1. To utilize the 'description' and 'name' feature in training, we first need to transform each description into a vector and then discover the relationship between the descriptions and the log price. We use BOW and TF-IDF techniques, respectively, during the transformation process. We initially train the transformed vectors with the log price using a linear regression model, so that the model's theta learns the potential relationship between the description vector and the log price. We extract the theta of the LR model based on the words in each description. Then, we sum up the values of the corresponding theta and append the result as a new feature to our final training set.

4. Sentiment
   1.  It calculates the sentiment scores for the description and the name of a datapoint by summing up the sentiment scores of each word in the cleaned 'description' and 'name' respectively. If a word is not found in the sentiment_dict, its sentiment score is considered as 0.
         ```python
         punctuation = set(string.punctuation)

         def sentiment(d):
            sentimentScore = 0
            r = ''.join([c for c in d.lower() if not c in punctuation])
            for w in r.split():
               sentimentScore += sentiment_dict.get(w, 0)
            return sentimentScore
         def name(d):
            sentimentScore = 0
            r = ''.join([c for c in d.lower() if not c in punctuation])
            for w in r.split():
               sentimentScore += name_dict.get(w, 0)
            return sentimentScore
         ```

5. Fix perfect multicollinearity(not incorporated into our model)
   1. Perfect multicollinearity happens when one variable can be perfectly predicted from the others, causing issues in regression models by inflating the variance of the coefficient estimates, which can lead to a very large MSE. By setting drop_first=True, the function will drop the first level for each categorical variable. This effectively removes one dummy variable from each set of dummies derived from a categorical variable, thus eliminating the perfect multicollinearity that occurs when all dummy variables for a category are included.
      ```python
      df_encoded = pd.get_dummies(df, columns=['cleaning_fee','host_has_profile_pic', 'host_identity_verified', 'instant_bookable'], drop_first=True)
      ```

6. Encoding
   1. One-Hot Encoding(not incorporated into our model):
      1. Categorical variables such as 'property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', and 'instant_bookable' were initially one-hot encoded. This process transforms categorical variables into a format that can be used in machine learning algorithms, creating separate binary columns for each category.
   2. KFold Target encoding:
      1. Due to the high dimensionality encountered with one-hot encoding, we implemented K-Fold target encoding to mitigate the issue and **we will use this encoding method to evaluate our models**. Target encoding is especially beneficial for neural network models. Where categorical features are replaced with the mean value of a target variable ('log_price') computed from each fold of the training data, to prevent data leakage.
   3. Leave One Out (LOO):
      1. Leave-One-Out (LOO) encoding is a form of target encoding that reduces overfitting by excluding the target value of the current row when calculating the category's mean target, thereby offering a more generalizable feature representation.
      <!-- 2. We tried LOO after target encoding, but obtained a MSE of 0.003 with our final model, which was dramastically lower than the prior MSE. We attemptted to find the reason that caused the reduction of the MSE but failed. Therefore, we decided not to ultilize this encoding technique until we find the reason. -->
   
7. Norm & Standard

We decided to use both standardization and normalization in our project. Standardization is great at handling the outliers we found in our data, which could really mess up our results if ignored. By combining both methods, we get the best of both worlds: easy-to-understand data from normalization and outlier resistance from standardization, making our data prep more effective and reliable.


### Model 1: 2nd degree Polynomial Regression
- Second-degree polynomial regression extends linear regression by modeling the relationship between the independent variable $x$ and the dependent variable $y$ as a quadratic equation of the form $y = ax^2 + bx + c$. This allows for capturing non-linear relationships between the variables, making it suitable for datasets where the trend bends or curves, rather than following a straight line.
   ```python
   degree=2
   model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
   model.fit(X_train, y_train)
   y_val_pred = model.predict(X_val)
   y_test_pred = model.predict(X_test)
   ```

### Model 2: Deep Neural Network (DNN)
- We build our second model with DNN, and train with hyperparameter tuner. 
   ```python
   def build_hp_model(hp):
      model = Sequential()
      # Iterate over the number of layers
      for i in range(hp.Int('num_layers', 2, 6)):
         model.add(Dense(units=hp.Int('units_' + str(i), min_value=16, max_value=96, step=16),
                           activation=hp.Choice('activation_' + str(i), ['leaky_relu'])))
      
      model.add(Dense(1))  # Output layer for regression
      learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
      
      model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                     loss='mean_squared_error',
                     metrics=['mean_squared_error'])
      return model
   
   tuner = keras_tuner.RandomSearch(
    hypermodel=build_hp_model,
    objective='val_loss',
    max_trials=8,
    seed=10,
    executions_per_trial=3,
    directory='tuner_results',
    project_name='keras_tuner_demo',
    overwrite=True
   )
   ```
- Two Keras callbacks are utilized to enhance the training process:

  - Early Stopping: Monitors the validation loss and stops training if there hasn't been a significant decrease (less than 0.001) in the validation loss for 5 consecutive epochs. This prevents overfitting and ensures the model restores the weights from the epoch with the best performance.
      ```
      early_stopping = keras.callbacks.EarlyStopping(
         monitor='val_loss',
         min_delta=0.001,
         patience=5,
         mode='min',
         restore_best_weights=True,
      )
      ```

  - Model Checkpoint: Saves the model at the filepath 'checkpoints' whenever a lower validation loss is observed. This ensures that the model configuration with the best validation performance is preserved, even if the model's performance degrades in subsequent epochs.
      ```
      model_checkpoint = keras.callbacks.ModelCheckpoint(
         filepath='best_model.h5',
         monitor='val_loss',
         save_best_only=True,
         save_weights_only= False,
         mode='min'
      )
      ```

<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/loss.jpg" alt="Figure 5" width="70%">
</div>
The two graphs describe the progression of loss metrics over successive epochs for a baseline neural network model and an hyperparameter tuned version. Each graph tracks how the loss on the training, validation, and test data sets evolves with each epoch. The green dot identifies the epoch at which the validation loss is minimized, suggesting the most effective model prior to any overfitting.

### Model 3: Extreme Gradient Boosting (XGBoost)
- We build our third model with XGBoost.
- This model employs the XGBoost framework to optimize a regression task, using a `GridSearchCV` to fine-tune hyperparameters over a specified grid. The key parameters include tree depth, learning rate, subsample rate, and feature sampling rate. The `XGBRegressor` is configured to minimize squared error with early stopping to prevent overfitting. The grid search explores combinations of these hyperparameters across a training dataset, evaluating performance through cross-validation to select the best model based on the negative mean squared error.
  
   ```python
   param_grid = {
   'max_depth': [3, 4, 5, 6, 7],
   'eta': [0.01, 0.05, 0.1, 0.2],
   'subsample': [0.6, 0.7, 0.8, 0.9],
   'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
   }
   xgb_model = xgb.XGBRegressor(n_estimators=250, random_state=42,objective='reg:squarederror', eval_metric='rmse',early_stopping_rounds=10)

   grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=False, n_jobs=-1)

   grid_search.fit(X_train_full, y_train_full, eval_set=[(X_val, y_val)], verbose=False)
   ```


## Result
- To evaluate our model, we utilized several metrics: 
  - Mean Squared Error (MSE) is favored for its emphasis on large errors
  - Mean Absolute Error (MAE) for its robustness to outliers and interpretability
  - Root Mean Squared Error (RMSE) for error representation in the target's units,
  - R-squared for indicating the model’s explanatory power in terms of the variance in the dependent variable it can predict. 
- Each metric provides unique insights, helping us understand different aspects of the model's performance.


The image showcases two types of plots(Residual Plot and Prediction Error Plot), each representing model evaluation for three different machine learning models: Polynomial Regression (Poly Reg), Neural Network (NN), and XGBoost (XGB).
<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/residual_kfold.jpg" alt="Figure 5" width="90%">
</div>

The image showcases two types of plots, each representing model evaluation for three different machine learning models: Polynomial Regression (Poly Reg), Neural Network (NN), and XGBoost (XGB).

The spread of data points indicates that all three models have errors that increase as the predicted values get higher, which could suggest a systematic bias in model predictions for higher values. There's no clear distinction between the models in terms of performance from these plots alone. They appear to perform similarly across the range of values. Maybe the data points with  yellow are a bit more concentrated, this bunching might hint that the XGBoost is doing a tad better at predictions compare to other two. 



<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/kfold_result.jpg" alt="Figure 6" width="70%">
</div>

Figure above: Performance comparison of three predictive models using standard evaluation metrics. The polynomial regression model, neural network, and XGBoost are evaluated by Mean Squared Error (MSE, blue), Root Mean Squared Error (RMSE, orange), Mean Absolute Error (MAE, green), and R-squared (R2, red). The values indicate the model's accuracy and error rate, with higher R2 values and lower error metric values generally indicating better performance. XGBoost performs slightly better than other models at all values.

## Discussion
Our analytical began with the adoption of polynomial regression due to its lower MSE than a linear model, suggesting a better fit. Mindful of the potential for overfitting with increased complexity, we transitioned to experimenting with DNNs to capitalize on their flexibility in modeling non-linearities. Our exploration culminated with the introduction of XGBoost, leveraging its ensemble learning capabilities to optimize our predictive performance.


In the process of refining our models, a crucial stage was the tuning of hyperparameters, especially for the neural network and XGBoost. For the neural network, we settled on the leaky ReLU activation function after observing its superior performance compared to the standard ReLU. For our XGBoost model, hyperparameter tuning focused on a range of values for 'max_depth', 'eta', 'subsample', and 'colsample_bytree' to fine-tune the model's complexity and learning rate, as well as its sampling methods. These parameters were chosen to balance the model's ability to learn from the data against the risk of overfitting.

We experimented with various categorical encoding techniques and made an intriguing observation: the XGBoost model's mean squared error significantly improved when we applied leave-one-out encoding. The underlying cause for this enhancement is not immediately clear to us.

<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/residual_loo.jpg" alt="Figure 7" width="90%">
</div>
Residuals for XGBoost appear to be more tightly clustered around the zero line, and the data points on prediction error plot are distributed around the diagonal (representing perfect predictions)
<div style="display: flex; justify-content: center;" align="center">
  <img src="graphs/loo_comp.jpg" alt="Figure 8" width="70%">
</div>
XGBoost has a very low MSE, RMSE, and MAE close to 0, which suggests excellent predictive accuracy. XGBoost achieves a perfect score of 1 for R2 score, which in real-world scenarios might suggest overfitting or a data leakage issue, as it's unusual for models to explain 100% of the variance in the target variable.

However, due to the perfect R^2 score,  and further investigation is warranted to ensure the results are valid and the model is generalizing well without overfitting.


## Conclusion
Reflecting on our project, there are several aspects where we could have approached things differently, such as data preprocessing and model selection.

Firstly, in terms of data preprocessing and feature engineering, while we employed various techniques such as dropping columns with excessive unique categories, handling missing values, and encoding categorical variables, we could use K-nearest neighbors (KNN) to impute missing values based on the values of the nearest neighbors in the feature space, and Principal Component Analysis (PCA) to reduce and compress the dimensionality of the feature space.

Additionally, our model selection process could have been more exhaustive. While we experimented with polynomial regression, DNNs, and XGBoost, future iterations of this project could involve conducting a more comprehensive model such as random forest regression, etc.

