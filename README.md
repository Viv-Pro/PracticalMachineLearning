---
title: "Practical Machine Learning Course Project"
author: "Vivaan Nanda"
date: "`r Sys.Date()`"
output: html_document
---

## Background

With the rise of wearable devices such as fitness trackers, it is possible to collect large amounts of data related to human activity. This project uses data from accelerometers on the belt, forearm, arm, and dumbbell of six participants to predict the quality of their barbell lifts. The outcome classe categorizes the movement into one correct execution (A) and four common mistakes (B–E).

---

## Data Loading

We import the datasets while explicitly converting empty strings and Excel division errors into NA values to ensure proper column typing.

```{r}
library(caret)
library(randomForest)

train_raw <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test_raw  <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```

## Data Cleaning

We remove variables with more than 95% missing values and exclude metadata (timestamps, window markers, and names) that do not contribute to the physical motion analysis. Finally, we ensure the outcome is treated as a factor for classification.

``` {r}
na_threshold <- 0.95 * nrow(train_raw)
cols_to_keep <- colSums(is.na(train_raw)) < na_threshold

training_clean <- train_raw[, cols_to_keep]
testing_clean  <- test_raw[, cols_to_keep]

training_clean <- training_clean[, -(1:7)]
testing_clean  <- testing_clean[, -(1:7)]

training_clean$classe <- as.factor(training_clean$classe)

final_features <- names(training_clean)[names(training_clean) != "classe"]
testing_clean  <- testing_clean[, final_features]
```

## Model Selection and Training

A Random Forest model was chosen for its high accuracy and ability to handle the non-linear nature of accelerometer data. To balance computational efficiency with model robustness, we used 3-fold cross-validation.

### Cross-Validation
```{r}
set.seed(123)
control <- trainControl(
  method = "cv",
  number = 3
)
```

### Training the Model
``` {r}
rf_model <- train(
  classe ~ .,
  data = training_clean,
  method = "rf",
  trControl = control,
  ntree=50
)

rf_model
```

## Results
The Random Forest model achieved very high accuracy during cross-validation, indicating strong predictive performance. This suggests that the model generalizes well to unseen data.

## Expected Out-of-Sample Error
The expected out-of-sample error is estimated during the cross-validation process. Since the cross-validation accuracy was approximately 99%, the expected out-of-sample error is estimated at 1% (1−accuracy). This suggests the model has excellent generalizability.

## Predictions on the Test Dataset
The model is applied to the 20 test cases to predict the activity class for the project submission.
```{r}
test_predictions <- predict(rf_model, testing_clean)
test_predictions
```

## Reproducibility
This analysis is fully reproducible. By setting a seed and using the caret package framework, the results can be replicated on any machine with the original datasets.
