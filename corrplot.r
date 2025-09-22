# Required libraries
library(corrplot)
library(xgboost)
library(caTools)
library(dplyr)
library(caret)

# Load data
data <- read.csv("Visitation Per Day.csv")

# Add column names
colnames(data) <- c("Visitation", "LogVisitation", "Date", "Day", "Max Temp", "Rainfall", "Solar", "Day of week", "WeekDayBool", "WeekendBool", "SchoolHolidayBool", "SeasonBool","Avg Temp","Temp Ratio","AvgSolar","SolarRatio")

# Remove NAs
data <- na.omit(data)

# Remove Date and Day columns
data <- data[, c(-3, -4)]

# Get labels (Visitation and LogVisitation)
labels <- data[, 1]
labelsLog <- data[, 2]

# Generate correlation matrix
M <- cor(data)
corrplot(M, method = 'number')

print(data[1,])
# Now we remove the Temp, Solar and AvgTemp and Avg Solar columns.
data <- data[,c(-3,-5,-11,-13)]
# Split data into training and testing
print(data[1,])
set.seed(123)  # For reproducibility
select <- sample(c(TRUE, FALSE), nrow(data), replace = TRUE, prob = c(0.8, 0.2))

# Training and testing sets for features
X_train <- na.omit(data[select, c(-1, -2)])
X_test <- na.omit(data[!select, c(-1, -2)])

# Training and testing sets for labels
y_train <- labels[select]
y_test <- labels[!select]

y_train_log <- labelsLog[select]
y_test_log <- labelsLog[!select]

# Create DMatrix for xgboost
xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_log)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test), label = y_test_log)

# Set parameters for regression
params <- list(
  objective = "reg:squarederror",  # Use squared error for regression
  eval_metric = "rmse"             # Root Mean Square Error
)

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 100,
  # evals = list(val = xgb_test),  # List of evaluation sets
  # early_stopping_rounds = 10,  # Stop if no improvement for 10 rounds
  verbose = 1
)

# Make predictions on test data
predictions <- predict(xgb_model, xgb_test)

# Print results
# Check the feature names of the training set used for training the model
feature_names <- colnames(X_train)

# Manually input one day of data
single_day_data <- data.frame(
  Rainfall = 0,   
  `Day of week` = 6,  
  WeekDayBool = 0,         
  WeekendBool = 1,         
  SchoolHolidayBool = 0,   
  SeasonBool = 4,
  TempRatio = 1,
  SolarRatio = 1

)

# Ensure the feature names in `single_day_data` match exactly with the training data
# Assign the same column names as used in the training data
colnames(single_day_data) <- feature_names

# Convert the single row into a matrix
single_day_matrix <- as.matrix(single_day_data)

#Create a DMatrix for the single-day prediction
xgb_single_day <- xgb.DMatrix(data = single_day_matrix)

# Make prediction using the trained model
single_day_prediction <- predict(xgb_model, xgb_single_day)

# Print the predicted visitation/log visitation
print(exp(single_day_prediction))

# RMSE (Root Mean Square Error)
rmse <- sqrt(mean((y_test_log - predictions)^2))

# MAE (Mean Absolute Error)
mae <- mean(abs(y_test_log - predictions))

# R-squared (Coefficient of determination)
sst <- sum((y_test_log - mean(y_test_log))^2)  # Total sum of squares
sse <- sum((y_test_log - predictions)^2)       # Sum of squared errors
r_squared <- 1 - (sse / sst)

# Print results
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("R-squared:", r_squared, "\n")


