# =============================================================================
# Lecture 2 — Classical Text Classification
# Practical Session: Classifying UK House of Commons Speeches
#
# Methods Seminar: Text Analysis for Political Science
# =============================================================================
#
# In this practical we will:
#   1. Reuse the UK Commons corpus from Lecture 1
#   2. Define a binary classification task: Government vs. Opposition
#   3. Build feature matrices (raw counts + TF-IDF)
#   4. Train Naive Bayes and Logistic Regression classifiers
#   5. Evaluate with accuracy, precision, recall, F1, confusion matrix
#   6. Inspect the most predictive words
#   7. Compare models and feature representations
#
# Prerequisites:
#   - The ParlSpeech V2 UK Commons file (Corp_HouseOfCommons_V2.rds)
#   - Packages installed (see section 0)
# =============================================================================


# -----------------------------------------------------------------------------
# 0. SETUP
# -----------------------------------------------------------------------------

# Run ONCE to install (comment out after first run)
# install.packages(c(
#   "quanteda",
#   "quanteda.textstats",
#   "quanteda.textmodels",
#   "glmnet",
#   "caret",
#   "tidyverse"
# ))

library(quanteda)
library(quanteda.textstats)
library(quanteda.textmodels)
library(glmnet)
library(caret)
library(tidyverse)

theme_set(theme_minimal(base_size = 12))
set.seed(42)  # For reproducibility — always set a seed!


# -----------------------------------------------------------------------------
# 1. LOAD AND PREPARE THE DATA
# -----------------------------------------------------------------------------

# Load ParlSpeech V2 (same file as Lecture 1)
speeches_raw <- readRDS("C:/Users/my27n/OneDrive/Рабочий стол/Humboldt_Fellowship/LLM_course/Lecture#1/Corp_HouseOfCommons_V2.rds")

# Filter to 2017-2019 parliament, keep substantive speeches
speeches <- speeches_raw %>%
  filter(date >= "2017-06-08", date <= "2019-11-06") %>%
  filter(terms >= 50) %>%
  filter(!is.na(party), party != "")

# --- Define our classification task ---
# Binary task: Government (Conservative) vs. Opposition (all others)
# This is a clean, politically meaningful distinction with clear expectations.
#
# Why this task?
# - Government and opposition MPs speak differently: government defends,
#   opposition attacks. This should be detectable from word choice alone.
# - The labels come from parliamentary metadata (no hand-coding needed).
# - It's a real political science question: how distinct is government
#   vs. opposition rhetoric?

speeches <- speeches %>%
  filter(party %in% c("Con", "Lab", "LibDem",
                       "SNP")) %>%
  mutate(label = ifelse(party == "Con", "Government", "Opposition"))

cat("Class distribution:\n")
print(table(speeches$label))
cat("\nProportion:\n")
print(prop.table(table(speeches$label)))

# Take a manageable sample (5,000 speeches — runs fast on any laptop)
speeches <- speeches %>% slice_sample(n = 5000)
cat("\nSample class distribution:\n")
print(table(speeches$label))


# -----------------------------------------------------------------------------
# 2. PREPROCESSING AND DFM
# -----------------------------------------------------------------------------

# Build corpus
corp <- corpus(speeches, text_field = "text")

# Tokenize and preprocess (same pipeline as Lecture 1)
toks <- tokens(corp,
               remove_punct   = TRUE,
               remove_numbers = TRUE,
               remove_url     = TRUE,
               remove_symbols = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_remove(c("hon", "friend", "gentleman", "lady", "member",
                   "members", "house", "speaker", "right"))

# Build DFM and trim
dfm_all <- dfm(toks) %>%
  dfm_trim(min_docfreq = 10,  docfreq_type = "count") %>%
  dfm_trim(max_docfreq = 0.5, docfreq_type = "prop")

cat("\nDFM dimensions:", ndoc(dfm_all), "documents x",
    nfeat(dfm_all), "features\n")
cat("Sparsity:", round(sparsity(dfm_all) * 100, 1), "%\n")


# -----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

# Stratified split: 80% train, 20% test
# Stratified = both classes have the same proportions in train and test

n <- ndoc(dfm_all)
labels <- docvars(dfm_all, "label")

# Create stratified indices
train_idx <- createDataPartition(labels, p = 0.8, list = FALSE)# Sample 80% of row indices, stratified by label (keeps class balance)
test_idx  <- setdiff(1:n, train_idx)# Remaining 20% of indices become the test set
# Split the DFM
dfm_train <- as.dfm(dfm_all[train_idx, ])
dfm_test  <- as.dfm(dfm_all[test_idx, ])

# CRITICAL: make sure test DFM has exactly the same features as train
# (new words in test that weren't in training must be dropped)
dfm_test  <- dfm_match(dfm_test, features = featnames(dfm_train))

cat("\nTraining set:", ndoc(dfm_train), "documents\n")
print(table(docvars(dfm_train, "label")))
cat("\nTest set:", ndoc(dfm_test), "documents\n")
print(table(docvars(dfm_test, "label")))


# -----------------------------------------------------------------------------
# 4. MODEL 1: NAIVE BAYES
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  MODEL 1: NAIVE BAYES\n")
cat("========================================\n\n")

# Train Naive Bayes (with Laplacian smoothing, built into quanteda)
# This is extremely fast — milliseconds even on large corpora
nb_model <- textmodel_nb(dfm_train, docvars(dfm_train, "label"))

# Predict on test set
nb_pred <- predict(nb_model, newdata = dfm_test)

# Confusion matrix and metrics
nb_cm <- confusionMatrix(
  nb_pred,
  factor(docvars(dfm_test, "label")),
  positive = "Opposition"  # define which class is "positive" for P/R/F1
)
print(nb_cm)

# Extract key metrics
cat("\n--- Naive Bayes Summary ---\n")
cat("Accuracy: ", round(nb_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(nb_cm$byClass["Precision"], 4), "\n")
cat("Recall:   ", round(nb_cm$byClass["Recall"], 4), "\n")
cat("F1:       ", round(nb_cm$byClass["F1"], 4), "\n")


# -----------------------------------------------------------------------------
# 5. MODEL 2: LOGISTIC REGRESSION (RIDGE)
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  MODEL 2: LOGISTIC REGRESSION (RIDGE)\n")
cat("========================================\n\n")

# glmnet needs a standard matrix (not a quanteda dfm object)
x_train <- as.matrix(dfm_train)
y_train <- docvars(dfm_train, "label")

# Train L2-regularized (Ridge) logistic regression
# cv.glmnet automatically finds the best regularization strength (lambda)
# using 5-fold cross-validation on the training set
lr_model <- cv.glmnet(
  x          = x_train,
  y          = y_train,
  family     = "binomial",   # logistic regression (binary classification)
  alpha      = 0,            # 0 = Ridge (L2), 1 = Lasso (L1)
  nfolds     = 5,            # 5-fold cross-validation for lambda
  type.measure = "class"     # optimize for classification accuracy
)

# Plot: how classification error varies with lambda
plot(lr_model, main = "Ridge LR: Classification Error vs. Regularization")

# The best lambda (least cross-validation error)
cat("Best lambda:", lr_model$lambda.min, "\n")

# Predict on test set
x_test <- as.matrix(dfm_test)
lr_pred <- predict(lr_model, x_test, s = "lambda.min", type = "class")

# Confusion matrix and metrics
lr_cm <- confusionMatrix(
  factor(lr_pred),
  factor(docvars(dfm_test, "label")),
  positive = "Opposition"
)
print(lr_cm)

cat("\n--- Logistic Regression (Ridge) Summary ---\n")
cat("Accuracy: ", round(lr_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(lr_cm$byClass["Precision"], 4), "\n")
cat("Recall:   ", round(lr_cm$byClass["Recall"], 4), "\n")
cat("F1:       ", round(lr_cm$byClass["F1"], 4), "\n")


# -----------------------------------------------------------------------------
# 6. MODEL 3: LOGISTIC REGRESSION (LASSO)
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  MODEL 3: LOGISTIC REGRESSION (LASSO)\n")
cat("========================================\n\n")

# Lasso (L1) pushes many weights to exactly zero = automatic feature selection
lr_lasso <- cv.glmnet(
  x          = x_train,
  y          = y_train,
  family     = "binomial",
  alpha      = 1,            # 1 = Lasso (L1)
  nfolds     = 5,
  type.measure = "class"
)

# How many features does Lasso keep (non-zero weights)?
lasso_coefs <- coef(lr_lasso, s = "lambda.min")
n_nonzero <- sum(lasso_coefs[-1, ] != 0)  # exclude intercept
cat("Lasso keeps", n_nonzero, "out of", nfeat(dfm_train), "features\n")

# Predict and evaluate
lr_lasso_pred <- predict(lr_lasso, x_test, s = "lambda.min", type = "class")

lr_lasso_cm <- confusionMatrix(
  factor(lr_lasso_pred),
  factor(docvars(dfm_test, "label")),
  positive = "Opposition"
)

cat("\n--- Logistic Regression (Lasso) Summary ---\n")
cat("Accuracy: ", round(lr_lasso_cm$overall["Accuracy"], 4), "\n")
cat("Precision:", round(lr_lasso_cm$byClass["Precision"], 4), "\n")
cat("Recall:   ", round(lr_lasso_cm$byClass["Recall"], 4), "\n")
cat("F1:       ", round(lr_lasso_cm$byClass["F1"], 4), "\n")


# -----------------------------------------------------------------------------
# 7. INTERPRETING THE WEIGHTS
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  INTERPRETING LOGISTIC REGRESSION WEIGHTS\n")
cat("========================================\n\n")

# Extract coefficients from the Ridge model
coefs <- coef(lr_model, s = "lambda.min")
coef_df <- data.frame(
  word   = rownames(coefs),
  weight = as.numeric(coefs[, 1])
) %>%
  filter(word != "(Intercept)", weight != 0) %>%
  arrange(desc(abs(weight)))

# Top words predictive of GOVERNMENT (positive weights)
# Note: glmnet codes the classes alphabetically.
# "Government" comes before "Opposition", so:
#   - positive weight = predicts Government
#   - negative weight = predicts Opposition
# VERIFY by checking which class glmnet treats as the reference:
cat("glmnet class levels:", paste(sort(unique(y_train)), collapse=", "), "\n")
cat("Positive weight = predicts the SECOND class alphabetically\n\n")
# In this case: Government < Opposition alphabetically
# So positive weight = predicts Opposition, negative = predicts Government
# (glmnet models P(second class) — here, P(Opposition))

top_opposition <- coef_df %>%
  filter(weight > 0) %>%
  head(20)

top_government <- coef_df %>%
  filter(weight < 0) %>%
  arrange(weight) %>%
  head(20)

cat("=== Top 20 words predicting OPPOSITION ===\n")
print(top_opposition, row.names = FALSE)

cat("\n=== Top 20 words predicting GOVERNMENT ===\n")
print(top_government, row.names = FALSE)

# Visualize the top predictive words
plot_data <- bind_rows(
  top_government %>% head(15) %>% mutate(direction = "Government"),
  top_opposition %>% head(15) %>% mutate(direction = "Opposition")
)

ggplot(plot_data, aes(x = reorder(word, weight), y = weight,
                       fill = direction)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Government" = "#0087DC",
                                "Opposition" = "#E4003B")) +
  labs(
    title = "Most Predictive Words: Government vs. Opposition",
    subtitle = "Logistic Regression (Ridge) coefficients",
    x = NULL,
    y = "Weight (negative = Government, positive = Opposition)",
    fill = NULL
  ) +
  theme(legend.position = "top")


# -----------------------------------------------------------------------------
# 8. SIDE-BY-SIDE MODEL COMPARISON
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  MODEL COMPARISON\n")
cat("========================================\n\n")

comparison <- data.frame(
  Model = c("Naive Bayes", "Logistic Reg. (Ridge)", "Logistic Reg. (Lasso)"),
  Accuracy  = c(nb_cm$overall["Accuracy"],
                lr_cm$overall["Accuracy"],
                lr_lasso_cm$overall["Accuracy"]),
  Precision = c(nb_cm$byClass["Precision"],
                lr_cm$byClass["Precision"],
                lr_lasso_cm$byClass["Precision"]),
  Recall    = c(nb_cm$byClass["Recall"],
                lr_cm$byClass["Recall"],
                lr_lasso_cm$byClass["Recall"]),
  F1        = c(nb_cm$byClass["F1"],
                lr_cm$byClass["F1"],
                lr_lasso_cm$byClass["F1"])
)
rownames(comparison) <- NULL
print(round(comparison[, -1], 4) %>% cbind(Model = comparison$Model, .))

# Visualize the comparison
comparison_long <- comparison %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("Naive Bayes" = "#5C7EA3",
                                "Logistic Reg. (Ridge)" = "#1B2A4A",
                                "Logistic Reg. (Lasso)" = "#D4763C")) +
  labs(
    title = "Model Comparison: Government vs. Opposition Classification",
    subtitle = "UK House of Commons, 2017–2019",
    y = "Score",
    x = NULL,
    fill = NULL
  ) +
  ylim(0, 1) +
  theme(legend.position = "top")


# -----------------------------------------------------------------------------
# 9. EXPERIMENT: TF-IDF vs. RAW COUNTS
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  EXPERIMENT: TF-IDF vs. RAW COUNTS\n")
cat("========================================\n\n")

# So far we used raw counts. Let's try TF-IDF and see if it helps.

# Create TF-IDF weighted DFM
dfm_tfidf_all   <- dfm_tfidf(dfm_all)
dfm_tfidf_train <- dfm_tfidf_all[train_idx, ]
dfm_tfidf_test  <- dfm_tfidf_all[test_idx, ]
dfm_tfidf_test  <- dfm_match(dfm_tfidf_test, featnames(dfm_tfidf_train))

# Naive Bayes with TF-IDF
nb_tfidf <- textmodel_nb(dfm_tfidf_train, docvars(dfm_tfidf_train, "label"))
nb_tfidf_pred <- predict(nb_tfidf, newdata = dfm_tfidf_test)
nb_tfidf_cm <- confusionMatrix(
  nb_tfidf_pred,
  factor(docvars(dfm_tfidf_test, "label")),
  positive = "Opposition"
)

# Logistic Regression (Ridge) with TF-IDF
lr_tfidf <- cv.glmnet(
  x      = as.matrix(dfm_tfidf_train),
  y      = docvars(dfm_tfidf_train, "label"),
  family = "binomial",
  alpha  = 0,
  nfolds = 5,
  type.measure = "class"
)
lr_tfidf_pred <- predict(lr_tfidf, as.matrix(dfm_tfidf_test),
                          s = "lambda.min", type = "class")
lr_tfidf_cm <- confusionMatrix(
  factor(lr_tfidf_pred),
  factor(docvars(dfm_tfidf_test, "label")),
  positive = "Opposition"
)

# Compare raw counts vs. TF-IDF
cat("=== Naive Bayes ===\n")
cat("  Raw counts F1:", round(nb_cm$byClass["F1"], 4), "\n")
cat("  TF-IDF F1:    ", round(nb_tfidf_cm$byClass["F1"], 4), "\n")

cat("\n=== Logistic Regression (Ridge) ===\n")
cat("  Raw counts F1:", round(lr_cm$byClass["F1"], 4), "\n")
cat("  TF-IDF F1:    ", round(lr_tfidf_cm$byClass["F1"], 4), "\n")


# -----------------------------------------------------------------------------
# 10. EXPERIMENT: DIFFERENT CLASSIFICATION TASKS
# -----------------------------------------------------------------------------

cat("\n========================================\n")
cat("  BONUS: MULTI-PARTY CLASSIFICATION\n")
cat("========================================\n\n")

# Instead of binary (Gov vs Opp), try classifying the actual party.
# This is a 4-class problem: Conservative, Labour, LibDem, SNP

# We need to rebuild the DFM with party as the label
dfm_party_train <- dfm_train
dfm_party_test  <- dfm_test
docvars(dfm_party_train, "party_label") <- docvars(dfm_party_train, "party")
docvars(dfm_party_test, "party_label")  <- docvars(dfm_party_test, "party")

# Naive Bayes handles multi-class naturally
nb_party <- textmodel_nb(dfm_party_train,
                          docvars(dfm_party_train, "party_label"))
nb_party_pred <- predict(nb_party, newdata = dfm_party_test)

# Confusion matrix for multi-class
nb_party_cm <- confusionMatrix(
  nb_party_pred,
  factor(docvars(dfm_party_test, "party_label"))
)

cat("Naive Bayes — 4-party classification:\n")
cat("Overall accuracy:", round(nb_party_cm$overall["Accuracy"], 4), "\n\n")
cat("Per-class F1 scores:\n")
print(round(nb_party_cm$byClass[, "F1"], 4))

cat("\nConfusion matrix:\n")
print(nb_party_cm$table)

# Discussion: which parties are easy to distinguish? Which are confused?
# SNP should be very distinctive (Scottish vocabulary).
# Labour vs. Conservative: distinguishable but harder.
# LibDem: often confused with Labour (similar vocabulary).

# For logistic regression multi-class: glmnet uses family = "multinomial"
lr_party <- cv.glmnet(
  x      = as.matrix(dfm_party_train),
  y      = docvars(dfm_party_train, "party_label"),
  family = "multinomial",
  alpha  = 0,
  nfolds = 5,
  type.measure = "class"
)

lr_party_pred <- predict(lr_party, as.matrix(dfm_party_test),
                          s = "lambda.min", type = "class")

lr_party_cm <- confusionMatrix(
  factor(lr_party_pred),
  factor(docvars(dfm_party_test, "party_label"))
)

cat("\nLogistic Regression — 4-party classification:\n")
cat("Overall accuracy:", round(lr_party_cm$overall["Accuracy"], 4), "\n\n")
cat("Per-class F1 scores:\n")
print(round(lr_party_cm$byClass[, "F1"], 4))


# =============================================================================
# DISCUSSION QUESTIONS
# =============================================================================
#
# 1. Which model performed better on the Gov vs. Opposition task?
#    Was the margin large or small? What does that tell you about the
#    difficulty of the task?
#
# 2. Look at the top predictive words (section 7). Do they make
#    substantive sense? Can you explain WHY these words distinguish
#    government from opposition speeches?
#
# 3. Did TF-IDF improve over raw counts? For which model?
#    Why might one representation work better than the other?
#
# 4. In the 4-party task (section 10), which parties were easiest
#    to classify? Which were most confused with each other? Why?
#
# 5. Try running the Lasso model (section 6) and check which words
#    it keeps vs. discards. Are the selected features interpretable?
#    How many features does the model need to perform well?
#
# =============================================================================
# HOMEWORK ASSIGNMENT
# =============================================================================
#
# Task: Build a text classifier for a political classification task.
#
# 1. Choose a task: Gov/Opp (as above), party classification,
#    policy topic, or your own idea with a different corpus.
#
# 2. Train BOTH Logistic Regression and Naive Bayes.
#
# 3. Report: accuracy, precision, recall, F1, and the confusion matrix.
#
# 4. Experiment with at least ONE variation:
#    - Raw counts vs. TF-IDF
#    - Ridge vs. Lasso
#    - With vs. without stop words
#    - Unigrams vs. unigrams + bigrams
#
# 5. Write a 1.5-page report: describe your task, results, which
#    model won, what the most predictive features reveal, and how
#    sensitive the results are to your choices.
#
# Submit: R script + report. Due before Lecture 3.
# =============================================================================
