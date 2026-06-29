# classification_machine_learning_project

This project is listed on my resume as "Admission Classification with ML."

Group ML classification project: predicting graduate business school admission outcomes (Admit / Borderline / Do Not Admit) from GPA and GMAT scores using 85 student records (admission.csv: GPA, GMAT, decision label, group).

Tools: R, MASS, caret, ggplot2

## Models Implemented
- **LDA** - linear decision boundaries
- **QDA** - quadratic decision boundaries
- **KNN** - k optimized by looping over all possible k values (1 to n_train), computing test error rate at each k, and selecting the minimum; optimal k identified with a plotted error curve
- Train/test split by stratified sampling: 5 test observations per class for balanced evaluation (group_by(De) %>% slice_head(n=5))
- Decision boundary plots generated on both training and test data for all three classifiers

## My Contributions
- LDA and QDA model implementation and evaluation on both training and test data
- get_perf_metrics() function: takes predicted probabilities, predicted classes, and ground truth; returns confusion matrix + results table with per-class sensitivity, per-class specificity, mean sensitivity, mean specificity, overall accuracy, overall error rate, and multiclass AUC, used across all three models (LDA, QDA, KNN)
- Co-authored written report and presented findings to the class

## Team Contributions
- EDA: scatter plot of GMAT vs. GPA colored by admission status; violin plots of GPA and GMAT by admission class
- KNN implementation with optimal-k selection and decision boundary plots on training and test data
- LDA and QDA decision boundary plots on training and test data (geom_raster background + geom_contour boundary lines)

## Deliverables
Dataset: admission.csv. R script (Admission Classification with ML.R), written report (Admission Classification with ML Report), and class presentation (Admission Classification with ML Presentation).
