# STAT387 Group 3 Final Project R Script

# Rylan Harrison
# Jocelyn Murguia Macias
# Kira Novitchkova-Burbank
# Andrin Zuellig

rm(list = ls())

# libraries
library(MASS) # For LDA and QDA
library(caret) # For KNN and Confusion Matrix
library(ggplot2) # For Plots
library(tidyverse) # For preprocessing
library(gridExtra) # For combining multiple plots
library(pROC) # For multiclass AUC


#=========================== Preprocessing ===========================#
admission <- read.csv("admission.csv")


str(admission)
admission$De <- as.factor(admission$De)

# Split into training and testing sets
admission_test <- admission %>% 
  group_by(De) %>% 
  slice_head(n=5)

admission_train <- anti_join(admission, admission_test)


#===================== Exploratory Data Analysis =====================#
summary(admission_train)

# Scatter plot of GMAT vs GPA
ggplot(admission_train, aes(x=GPA, y=GMAT, col=De)) +
  geom_point() + 
  scale_color_manual("Admission Status",values=c("green", 'orange', "red"),
                     labels=c("Admit", "Borderline", "Do Not Admit")) +
  xlab("GPA") +
  ylab("GMAT Score") +
  theme_bw()

# Violin plots: GPA and GMAT vs admission status 
create_violin_plot <- function(data, y_var, y_label=y_var) {
  ggplot(data, aes(x = De, y = .data[[y_var]], fill = De)) +
    geom_violin() +
    scale_fill_manual(values = c("green", "orange", "red")) +
    guides(fill = "none") +
    xlab("Admission Status") +
    ylab(y_label) +
    scale_x_discrete(labels = c("Admit", "Borderline", "Do Not Admit")) +
    theme_bw()
}

gmat_vs_status <- create_violin_plot(admission_train, "GMAT", "GMAT Score")
gpa_vs_status <- create_violin_plot(admission_train, "GPA")
grid.arrange(gmat_vs_status, gpa_vs_status, nrow = 1, ncol = 2)


#==================== Linear Discriminant Analysis ====================#
lda.fit <- lda(De ~ GPA + GMAT, data = admission_train)

# lda with training data
lda.pred_train <- predict(lda.fit, admission_train)
lda.class_train <- lda.pred_train$class

lda.probs_train <- lda.pred_train$posterior # probabilities for each class

# defining admission_train$De as lda.ground_truth_train for clarity
lda.ground_truth_train <- admission_train$De

# Automate extraction of performance metrics and confusion matrix
get_perf_metrics <- function(probs, pred, ground_truth){
  # Use this to get sensitivities and specificities
  cm <- confusionMatrix(data=pred, reference=ground_truth, positive="1")
  sensitivities <- cm$byClass[,"Sensitivity"]
  specificities <- cm$byClass[,"Specificity"]
  overall_accuracy <- cm$overall["Accuracy"]
  overall_error_rate <- 1 - overall_accuracy
  mean_sens <- mean(sensitivities)
  mean_spec <- mean(specificities)
  
  # Calculate the AUC
  auc <- multiclass.roc(ground_truth, probs)$auc
  
  # Put the results into a table
  result_vector <- c(sensitivities, mean_sens, 
                     specificities, mean_spec, 
                     overall_accuracy, overall_error_rate, 
                     auc)
  
  row_names <- c(paste0(names(sensitivities), " Sensitivity"),
                 "Mean Sensitivity",
                 paste0(names(specificities), " Specificity"),
                 "Mean Specificity",
                 "Overall Accuracy", 
                 "Overall Error Rate", 
                 "AUC")
  
  result_df <- data.frame(Metric = row_names, Value = result_vector)
  
  # Return confusion matrix and results table as list
  results <- list(Confusion_Matrix=cm$table, Metrics=result_df)
  return(results)
}

cat("LDA with Training Data: \n\n")

get_perf_metrics(lda.probs_train, lda.class_train, lda.ground_truth_train)



# lda with test data
lda.pred_test <- predict(lda.fit, admission_test)
lda.class_test <- lda.pred_test$class

lda.probs_test <- lda.pred_test$posterior # probabilities for each class

# defining admission_test$De as lda.ground_truth_test for clarity
lda.ground_truth_test <- admission_test$De

cat("LDA with Test Data: \n\n")

get_perf_metrics(lda.probs_test, lda.class_test, lda.ground_truth_test)


#================== Quadratic Discriminant Analysis ==================#
# qda with training data
qda.fit <- qda(De ~ GPA + GMAT, data = admission_train)
qda.pred_train <- predict(qda.fit, admission_train)
qda.class_train <- qda.pred_train$class

qda.probs_train <- qda.pred_train$posterior # probabilities for each class

# defining admission_train$De as qda.ground_truth_train for clarity
qda.ground_truth_train <- admission_train$De


cat("QDA with Training Data: \n\n")

get_perf_metrics(qda.probs_train, qda.class_train, qda.ground_truth_train)



# qda with test data
qda.pred_test <- predict(qda.fit, admission_test)
qda.class_test <- qda.pred_test$class

qda.probs_test <- qda.pred_test$posterior # probabilities for each class
# defining admission_test$De as qda.ground_truth_test for clarity
qda.ground_truth_test <- admission_test$De


cat("QDA with Test Data: \n\n")

get_perf_metrics(qda.probs_test, qda.class_test, qda.ground_truth_test)


#================== LDA and QDA Decision Boundaries ==================#

##### plot LDA desicion boundary line
GPA_range <- seq(min(admission_train$GPA) - 1, 
                 max(admission_train$GPA) + 1, length.out = 500)
GMAT_range <- seq(min(admission_train$GMAT) - 1,
                  max(admission_train$GMAT) + 1, length.out = 500)
grid <- expand.grid(GPA = GPA_range, GMAT = GMAT_range)

# Predict the class probabilities for the grid points
grid$De <- predict(lda.fit, newdata = grid)$class

# Create numeric labels for the class (1 = admit, 2 = notadmit, 3 = border)
grid$De_num <- as.numeric(factor(grid$De, 
                                 levels = c("admit", "notadmit", "border")))


#LDA desicion boundary line on training data
lda_plot_train <- ggplot(data = admission_train, aes(x = GPA, y = GMAT, 
                                                     color = De)) +
  geom_point() +
  # geom_raster for smooth background # alpha = 0.1 for transparent background
  geom_raster(data = grid, aes(x = GPA, y = GMAT, fill = De), alpha = 0.1) +  
  # Point colors
  scale_color_manual("Admission Status", 
                     values = c("admit" = "green",
                                "notadmit" = "red", "border" = "orange"),
                     labels = c("admit" = "Admit","border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  # Background colors
  scale_fill_manual("Classification",
                    values = c("admit" = "green",
                               "notadmit" = "red", "border" = "orange"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  theme_minimal() +
  # Remove grid lines
  theme(panel.grid = element_blank()) +
  # Add decision boundary lines
  geom_contour(data = grid, aes(x = GPA, y = GMAT, z = De_num), 
               color = "black", size = 0.2)


#LDA desicion boundary line on test data
lda_plot_test <- ggplot(data = admission_test, aes(x = GPA, y = GMAT, 
                                                   color = De)) +
  geom_point() +
  # geom_raster for smooth background # alpha = 0.1 for transparent background
  geom_raster(data = grid, aes(x = GPA, y = GMAT, fill = De), alpha = 0.1) +  
  # Point colors
  scale_color_manual("Admission Status", 
                     values = c("admit" = "green",
                                "notadmit" = "red", "border" = "orange"),
                     labels = c("admit" = "Admit", "border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  # Background colors
  scale_fill_manual("Classification",
                    values = c("admit" = "green",
                               "notadmit" = "red", "border" = "orange"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  theme_minimal() +
  # Remove grid lines
  theme(panel.grid = element_blank()) +
  # Add decision boundary lines
  geom_contour(data = grid, aes(x = GPA, y = GMAT, z = De_num), 
               color = "black", size = 0.2)

lda_plot_train
lda_plot_test



##### plot QDA desicion boundary line
grid2 <- expand.grid(GPA = GPA_range, GMAT = GMAT_range)

# Predict the class probabilities for the grid points
grid2$De <- predict(qda.fit, newdata = grid2)$class

# Create numeric labels for the class (1 = admit, 2 = notadmit, 3 = border)
grid2$De_num <- as.numeric(factor(grid2$De, levels = c("admit", "notadmit", 
                                                       "border")))


# QDA desicion boundary line on training data
qda_plot_train <- ggplot(data = admission_train, aes(x = GPA, y = GMAT,
                                                     color = De)) +
  geom_point() +
  # geom_raster for smooth background # alpha = 0.1 for transparent background
  geom_raster(data = grid2, aes(x = GPA, y = GMAT, fill = De), alpha = 0.1) +  
  # Point colors
  scale_color_manual("Admission Status", 
                     values = c("admit" = "green",
                                "notadmit" = "red", "border" = "orange"),
                     labels = c("admit" = "Admit", "border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  # Background colors
  scale_fill_manual("Classification",
                    values = c("admit" = "green",
                               "notadmit" = "red", "border" = "orange"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  theme_minimal() +
  # Remove grid lines
  theme(panel.grid = element_blank()) +
  # Add decision boundary lines
  geom_contour(data = grid2, aes(x = GPA, y = GMAT, z = De_num), 
               color = "black", size = 0.2)


# QDA desicion boundary line on test data
qda_plot_test <- ggplot(data = admission_test, aes(x = GPA, y = GMAT,
                                                   color = De)) +
  geom_point() +
  # geom_raster for smooth background # alpha = 0.1 for transparent background
  geom_raster(data = grid2, aes(x = GPA, y = GMAT, fill = De), alpha = 0.1) +  
  # Point colors
  scale_color_manual("Admission Status", 
                     values = c("admit" = "green",
                                "notadmit" = "red", "border" = "orange"),
                     labels = c("admit" = "Admit", "border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  # Background colors
  scale_fill_manual("Classification",
                    values = c("admit" = "green",
                               "notadmit" = "red", "border" = "orange"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  theme_minimal() +
  # Remove grid lines
  theme(panel.grid = element_blank()) +
  # Add decision boundary lines
  geom_contour(data = grid2, aes(x = GPA, y = GMAT, z = De_num), 
               color = "black", size = 0.2)


qda_plot_train
qda_plot_test



#======================== K-Nearest Neighbors ========================#
set.seed(70)

train_X <- as.matrix(admission_train[, c('GPA', 'GMAT')])
train_X <- scale(train_X)
train_Y <- admission_train$De

test_X <- as.matrix(admission_test[, c('GPA', 'GMAT')])
test_X <- scale(test_X)
test_Y <- admission_test$De

# Calculates error and predictions (can use on either train
# or test data, in case we want to plot train and test error rates together)
calc_knn_error <- function(num_neighbors, train_obs, test_obs, train_labels, 
                           test_labels){
  knn_model <- knn3(train_obs, train_labels, k = num_neighbors)
  knn_pred <- predict(knn_model, test_obs, type = "class")  
  err <- 1 - mean(test_labels == knn_pred)
  return(err)
}

test_error_rates <- numeric(nrow(admission_train))
for (i in 1:nrow(admission_train)) {
  test_error_rates[i] <- calc_knn_error(num_neighbors=i, train_obs = train_X, 
                                        test_obs=test_X, train_labels=train_Y, 
                                        test_labels=test_Y)
}

plot(test_error_rates, type='l', xlab='K', ylab='Test Error Rate', col='blue')
optimal_k <- which.min(test_error_rates)
points(optimal_k, test_error_rates[optimal_k], col='red', pch=16)
legend(0, 0.6, legend="Optimal K and Test Error", col='red', pch=16)

# Train k-NN model with optimal k
knn_model <- knn3(train_X, train_Y, k = optimal_k)

# Get predicted class probabilities
knn_train_probs <- predict(knn_model, train_X, type = "prob")
knn_test_probs <- predict(knn_model, test_X, type = "prob")

# Get predicted class labels
knn_train_out <- predict(knn_model, train_X, type = "class")
knn_test_out <- predict(knn_model, test_X, type = "class")

# Train Performance Metrics 
knn_train_metrics <- get_perf_metrics(knn_train_probs, knn_train_out, train_Y)
knn_train_metrics

# Test Performance Metrics
knn_test_metrics <- get_perf_metrics(knn_test_probs, knn_test_out, test_Y)
knn_test_metrics


#====================== KNN Decision Boundaries ======================#
grid_width <- 200
x1_grid <- seq(min(train_X[, 1]) - 0.5, max(train_X[, 1]) + 0.5, l=grid_width)
x2_grid <- seq(min(train_X[, 2]) - 0.5, max(train_X[, 2]) + 0.5, l=grid_width)
grid <- expand.grid(x1=x1_grid, x2=x2_grid)

grid$pred <- predict(knn_model, grid, type='class')

# For train set 
ggplot() + 
  geom_raster(data=grid, aes(x=x1, y=x2, fill=pred), alpha=0.1) +
  geom_point(data=train_X, aes(x=GPA, y=GMAT, color=train_Y), alpha=1) +
  scale_color_manual("Admission Status", 
                     values = c("notadmit" = "red", "border" = "orange", 
                                "admit" = "green"), 
                     labels = c("admit" = "Admit", "border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  scale_fill_manual("Classification", 
                    values = c("notadmit" = "red", "border" = "orange", 
                               "admit" = "green"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  scale_x_continuous(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) +
  labs(x = "Standardized GPA", y = "Standardized GMAT") +
  theme_minimal() +
  theme(panel.grid = element_blank(), text = element_text(size=13)) 


# For test set 
ggplot() + 
  geom_raster(data=grid, aes(x=x1, y=x2, fill=pred), alpha=0.1) +
  geom_point(data=test_X, aes(x=GPA, y=GMAT, color=test_Y), alpha=1) +
  scale_color_manual("Admission Status", 
                     values = c("notadmit" = "red", "border" = "orange", 
                                "admit" = "green"), 
                     labels = c("admit" = "Admit", "border" = "Borderline",
                                "notadmit" = "Do Not Admit")) +
  scale_fill_manual("Classification", 
                    values = c("notadmit" = "red", "border" = "orange", 
                               "admit" = "green"), 
                    labels = c("admit" = "Admit", "border" = "Borderline",
                               "notadmit" = "Do Not Admit")) +
  scale_x_continuous(expand=c(0, 0)) +
  scale_y_continuous(expand=c(0, 0)) +
  labs(x = "Standardized GPA", y = "Standardized GMAT") +
  theme_minimal() +
  theme(panel.grid = element_blank(), text = element_text(size=13)) 