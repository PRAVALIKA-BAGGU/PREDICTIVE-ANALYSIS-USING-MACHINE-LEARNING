# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING


*DESCRIPTION*:

This project involves building and evaluating a simple machine learning model using Python to understand how different variables can be used to predict an outcome. It starts by installing the required libraries such as scikit-learn, pandas, matplotlib, and seaborn, which are essential tools for data manipulation, machine learning, and visualization in Python. A synthetic dataset is created using pandas, containing two input features and a target variable. This dummy dataset helps simulate a real-world scenario without relying on external files, making the project self-contained and reproducible in any environment, including Google Colab.

After generating the dataset, the data is explored to understand its structure and statistical properties. The `info()` and `describe()` functions provide basic details about data types, missing values, and summary statistics like mean and standard deviation. To further understand the relationships within the dataset, visualizations are created using seaborn’s pairplot and a heatmap. These plots give insights into how each feature correlates with the target and with one another, which is crucial for making decisions about feature selection.

The next part of the project focuses on selecting the independent variables (features) and the dependent variable (target) for model training. In this example, both feature1 and feature2 are used to predict the target. The dataset is then split into training and testing subsets using scikit-learn’s train\_test\_split function. This ensures that the model is trained on one part of the data and tested on a different part, which helps evaluate how well the model generalizes to unseen data.

A linear regression model is chosen for this task as it is simple, interpretable, and effective for problems where the target variable is continuous. The model is trained using the training dataset by fitting it to the features and target. Once trained, the model is used to make predictions on the test dataset. These predictions are then compared to the actual values of the target variable to assess the model’s performance.

To evaluate how accurately the model has learned the relationships in the data, several metrics are calculated. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) measure the average prediction error, with RMSE being in the same unit as the target variable. The R-squared (R²) metric tells us how much of the variance in the target variable is explained by the model. A higher R² value indicates a better fit between the predicted and actual values.

Finally, a scatter plot is created to visually compare the predicted values against the actual values. A red diagonal reference line is added to show the ideal prediction line, where predicted equals actual. Points that lie close to this line indicate accurate predictions, while deviations from the line highlight errors. This visual representation is a quick and intuitive way to understand the model’s performance. Overall, this task provides hands-on experience in building a basic regression model, understanding data relationships, evaluating performance, and visualizing results, forming the foundation for more complex machine learning projects.

 *OUTPUT*:
 
