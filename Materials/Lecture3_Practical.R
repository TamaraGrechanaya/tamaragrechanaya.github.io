rm(list=ls(all=TRUE))
# =============================================================================
# Lecture 3 — Word Embeddings & Vector Spaces
# Practical Session: Exploring Pre-trained English Word Embeddings
#
# Methods Seminar: Multimodal Computational Methods for Political Science
# Computational Social Science Program, LMU Munich
# =============================================================================
#
# In this practical we will:
#   1. Load pre-trained English word embeddings (GloVe)
#   2. Find nearest neighbors of political concepts
#   3. Compute cosine similarities between word pairs
#   4. Test word analogies (king - man + woman = ?)
#   5. Visualize political vocabulary with PCA
#   6. Measure concept distances using seed-word axes
#   7. Explore embedding bias (WEAT-style analysis)
#
# Data:
#   We use GloVe pre-trained vectors (easier to load than fastText in R).
#   Download "glove.2024.wikigiga.100d.zip" from: https://nlp.stanford.edu/projects/glove/
#   Unzip and use "glove.2024.wikigiga.100d.zip" (100 dimensions, ~350 MB).
#   For better-quality analysis use "glove.2024.wikigiga.300d.zip" (300 dimensions, ~1 GB).

# Alternative (no download needed):
# The textdata package can download GloVe automatically — see section 1b.
# This option might generate an error (Stanford NLP server that hosts GloVe is slow and often unreliable)
# =============================================================================


# -----------------------------------------------------------------------------
# 0. SETUP
# -----------------------------------------------------------------------------

# Run ONCE to install (comment out after first run)
# install.packages(c(
#   "tidyverse",
#   "ggplot2",
#   "ggrepel",
#   "textdata",    # for automatic GloVe download
#   "text2vec"     # alternative: manual loading
# ))
library(tidyverse)
library(ggplot2)
library(ggrepel)

library(text2vec)
library(textdata)
set.seed(42)


# -----------------------------------------------------------------------------
# 1a. LOAD EMBEDDINGS: Manual method (if you downloaded glove.6B.zip)
# -----------------------------------------------------------------------------

# This function reads GloVe .txt format into a named matrix
glove_tbl <- read_delim(
  ".../wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt",
  delim = " ",
  quote = "",                  # GloVe uses raw spaces, no quotes
  col_names = c("token", paste0("d", 1:100)),
  show_col_types = FALSE
)

# Convert from tibble to a named matrix (rows = words, cols = dimensions)
emb <- as.matrix(glove_tbl[, -1])  # drop the "token" column
rownames(emb) <- glove_tbl$token

cat("Loaded", nrow(emb), "word vectors with", ncol(emb), "dimensions.\n")

# Check that a few words are present
c("democracy", "parliament", "climate") %in% rownames(emb)

# -----------------------------------------------------------------------------
# 1b. LOAD EMBEDDINGS: Automatic method (using textdata package)
# -----------------------------------------------------------------------------
# This downloads GloVe automatically on first run (~800 MB download)
# and caches it locally for future use.

library(textdata)

# Download and load GloVe 6B (trained on Wikipedia + Gigaword)
# First run will prompt you to confirm the download
glove_tbl <- embedding_glove6b(dimensions = 100)  # use 100d for speed
# For better quality, use: embedding_glove6b(dimensions = 300)

# Convert from tibble to a named matrix (rows = words, cols = dimensions)
emb <- as.matrix(glove_tbl[, -1])  # drop the "token" column
rownames(emb) <- glove_tbl$token

cat("Loaded", nrow(emb), "word vectors with", ncol(emb), "dimensions.\n")

# Check that a few words are present
c("democracy", "parliament", "climate") %in% rownames(emb)


# -----------------------------------------------------------------------------
# 2. CORE FUNCTIONS: Cosine Similarity & Nearest Neighbors
# -----------------------------------------------------------------------------

# Cosine similarity between two vectors
cosine_sim <- function(u, v) {
  sum(u * v) / (sqrt(sum(u^2)) * sqrt(sum(v^2)))
}

# Find the k nearest neighbors of a word
nearest_neighbors <- function(word, embeddings, k = 10) {
  if (!(word %in% rownames(embeddings))) {
    cat("Word '", word, "' not in vocabulary!\n")
    return(NULL)
  }

  target <- embeddings[word, ]

  # Compute cosine similarity to ALL words (vectorized for speed)
  # Dot products: embeddings %*% target gives a column vector
  dots  <- embeddings %*% target
  norms <- sqrt(rowSums(embeddings^2))
  target_norm <- sqrt(sum(target^2))
  sims  <- dots / (norms * target_norm)

  # Remove the word itself
  sims <- sims[rownames(embeddings) != word, , drop = FALSE]

  # Sort and return top k
  top_idx <- order(sims, decreasing = TRUE)[1:k]
  result <- data.frame(
    word = rownames(sims)[top_idx],
    similarity = round(sims[top_idx, 1], 4)
  )
  return(result)
}


# -----------------------------------------------------------------------------
# 3. EXPLORE: Nearest Neighbors of Political Concepts
# -----------------------------------------------------------------------------
# Try different political concepts — discuss what you see!
cat("\n--- democracy ---\n")
print(nearest_neighbors("democracy", emb))

cat("\n--- immigration ---\n")
print(nearest_neighbors("immigration", emb))

cat("\n--- welfare ---\n")
print(nearest_neighbors("welfare", emb))

cat("\n--- sovereignty ---\n")
print(nearest_neighbors("sovereignty", emb))

cat("\n--- austerity ---\n")
print(nearest_neighbors("austerity", emb))

cat("\n--- populism ---\n")
print(nearest_neighbors("populism", emb))

# Discussion: Do the neighbors make semantic sense?
# Are there any surprises? Any words you didn't expect?


# -----------------------------------------------------------------------------
# 4. COSINE SIMILARITY BETWEEN SPECIFIC WORD PAIRS
# -----------------------------------------------------------------------------
# Words we expect to be SIMILAR
cat("climate - environment:  ", cosine_sim(emb["climate",], emb["environment",]), "\n")
cat("election - vote:        ", cosine_sim(emb["election",], emb["vote",]), "\n")
cat("parliament - congress:  ", cosine_sim(emb["parliament",], emb["congress",]), "\n")
cat("poverty - inequality:   ", cosine_sim(emb["poverty",], emb["inequality",]), "\n")

# Words we expect to be DISSIMILAR
cat("climate - bicycle:      ", cosine_sim(emb["climate",], emb["bicycle",]), "\n")
cat("democracy - sandwich:   ", cosine_sim(emb["democracy",], emb["sandwich",]), "\n")
cat("parliament - banana:    ", cosine_sim(emb["parliament",], emb["banana",]), "\n")

# Politically interesting comparisons
cat("freedom - equality:     ", cosine_sim(emb["freedom",], emb["equality",]), "\n")
cat("freedom - security:     ", cosine_sim(emb["freedom",], emb["security",]), "\n")
cat("taxes - welfare:        ", cosine_sim(emb["taxes",], emb["welfare",]), "\n")
cat("immigration - crime:    ", cosine_sim(emb["immigration",], emb["crime",]), "\n")
cat("immigration - labor:    ", cosine_sim(emb["immigration",], emb["labor",]), "\n")

# Discussion: Is immigration closer to "crime" or to "labor"?
# What does this tell us about the training corpus (Wikipedia + news)?


# -----------------------------------------------------------------------------
# 5. SIMILARITY MATRIX: Political Concept Map
# -----------------------------------------------------------------------------

# Pick a set of concepts and compute all pairwise similarities
concepts <- c("democracy", "freedom", "equality", "justice",
              "security", "immigration", "economy", "welfare",
              "climate", "education", "military", "religion")

# Build the similarity matrix
sim_matrix <- matrix(0, nrow = length(concepts), ncol = length(concepts))
for (i in seq_along(concepts)) {
  for (j in seq_along(concepts)) {
    sim_matrix[i, j] <- cosine_sim(emb[concepts[i], ], emb[concepts[j], ])
  }
}
rownames(sim_matrix) <- concepts
colnames(sim_matrix) <- concepts

# Print (rounded for readability)
cat("\nPairwise cosine similarities:\n")
print(round(sim_matrix, 3))

# Heatmap visualization
sim_df <- as.data.frame(as.table(sim_matrix))
colnames(sim_df) <- c("Word1", "Word2", "Similarity")

ggplot(sim_df, aes(x = Word1, y = Word2, fill = Similarity)) +
  geom_tile() +
  geom_text(aes(label = round(Similarity, 2)), size = 2.5) +
  scale_fill_gradient2(low = "#E4003B", mid = "white", high = "#1B2A4A",
                        midpoint = 0.3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Cosine Similarity Between Political Concepts",
       subtitle = "GloVe embeddings (trained on Wikipedia + Gigaword)",
       x = NULL, y = NULL)


# -----------------------------------------------------------------------------
# 6. WORD ANALOGIES
# -----------------------------------------------------------------------------
# Solve: a is to b as c is to ?
# Compute: b - a + c, then find nearest neighbor
solve_analogy <- function(a, b, c, embeddings, k = 5) {
  # Check all words exist
  for (w in c(a, b, c)) {
    if (!(w %in% rownames(embeddings))) {
      cat("Word '", w, "' not in vocabulary!\n")
      return(NULL)
    }
  }

  # Compute the target vector
  target <- embeddings[b, ] - embeddings[a, ] + embeddings[c, ]

  # Cosine similarity to all words
  dots  <- embeddings %*% target
  norms <- sqrt(rowSums(embeddings^2))
  target_norm <- sqrt(sum(target^2))
  sims  <- dots / (norms * target_norm)

  # Exclude the input words
  exclude <- c(a, b, c)
  sims <- sims[!(rownames(embeddings) %in% exclude), , drop = FALSE]

  # Top k
  top_idx <- order(sims, decreasing = TRUE)[1:k]
  result <- data.frame(
    word = rownames(sims)[top_idx],
    similarity = round(sims[top_idx, 1], 4)
  )
  return(result)
}

# Classic analogies
cat("\nking - man + woman = ?\n")
print(solve_analogy("man", "king", "woman", emb))

cat("\nfrance - paris + london = ?\n")
print(solve_analogy("paris", "france", "london", emb))

cat("\nwalking - walked + swam = ?\n")
print(solve_analogy("walked", "walking", "swam", emb))

# Political analogies — these are more interesting (and less reliable)
cat("\ndemocrat - liberal + conservative = ?\n")
print(solve_analogy("liberal", "democrat", "conservative", emb))

cat("\nsenator - senate + parliament = ?\n")
print(solve_analogy("senate", "senator", "parliament", emb))

# Discussion:
# - Which analogies work well? Which fail?
# - Why do political analogies often fail?
#   (Hint: political relationships are more complex and culturally
#    specific than gender or capital-country mappings)


# -----------------------------------------------------------------------------
# 7. PCA VISUALIZATION
# -----------------------------------------------------------------------------
# Pick words from several political/thematic categories
words_to_plot <- c(
  # Climate / environment
  "climate", "environment", "pollution", "carbon", "renewable",
  # Economy
  "economy", "market", "taxes", "growth", "unemployment",
  # Social policy
  "welfare", "poverty", "inequality", "education", "healthcare",
  # Immigration / security
  "immigration", "border", "asylum", "refugee", "security",
  # Cities (control group — should cluster separately)
  "london", "paris", "berlin", "washington", "tokyo"
)

# Check which words are actually in the vocabulary
available <- words_to_plot[words_to_plot %in% rownames(emb)]
missing   <- words_to_plot[!words_to_plot %in% rownames(emb)]
if (length(missing) > 0) cat("Missing from vocabulary:", missing, "\n")

# Extract their vectors
vecs <- emb[available, ]

# Run PCA
pca_result <- prcomp(vecs, scale. = TRUE)

# How much variance do the first two components explain?
var_explained <- summary(pca_result)$importance[2, 1:2]
cat("Variance explained: PC1 =", round(var_explained[1]*100, 1),
    "%, PC2 =", round(var_explained[2]*100, 1), "%\n")

# Create plot data frame
plot_df <- data.frame(
  word = available,
  PC1  = pca_result$x[, 1],
  PC2  = pca_result$x[, 2]
)

# Add category labels for coloring
plot_df$category <- case_when(
  plot_df$word %in% c("climate","environment","pollution","carbon","renewable") ~ "Climate",
  plot_df$word %in% c("economy","market","taxes","growth","unemployment") ~ "Economy",
  plot_df$word %in% c("welfare","poverty","inequality","education","healthcare") ~ "Social Policy",
  plot_df$word %in% c("immigration","border","asylum","refugee","security") ~ "Immigration",
  plot_df$word %in% c("london","paris","berlin","washington","tokyo") ~ "Cities",
  TRUE ~ "Other"
)

# Plot
ggplot(plot_df, aes(x = PC1, y = PC2, color = category, label = word)) +
  geom_point(size = 3) +
  geom_text_repel(size = 3.5, max.overlaps = 20) +
  scale_color_manual(values = c(
    "Climate"       = "#4A9C2D",
    "Economy"       = "#0087DC",
    "Social Policy" = "#E4003B",
    "Immigration"   = "#D4763C",
    "Cities"        = "#6C3483"
  )) +
  labs(
    title = "Political Vocabulary in Embedding Space",
    subtitle = paste0("PCA projection (PC1: ", round(var_explained[1]*100,1),
                      "%, PC2: ", round(var_explained[2]*100,1), "%)"),
    x = "Principal Component 1",
    y = "Principal Component 2",
    color = "Category"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

# Discussion:
# - Do the categories form visible clusters?
# - Which words are close to each other across categories?
# - Where does "security" land — near immigration or near military?
# - Remember: this is a projection from 100/300 dims to 2.
#   Some distortion is inevitable.


# -----------------------------------------------------------------------------
# 8. CONCEPT AXES: Measuring Ideology with Embeddings
# -----------------------------------------------------------------------------
# Define a left-right axis using seed words
# These are words we associate with each pole of the spectrum
left_seeds  <- c("solidarity", "equality", "welfare", "workers",
                  "collective", "redistribution", "union")
right_seeds <- c("market", "competition", "enterprise", "individual",
                  "liberty", "privatization", "business")

# Check availability
left_available  <- left_seeds[left_seeds %in% rownames(emb)]
right_available <- right_seeds[right_seeds %in% rownames(emb)]

cat("Left seeds found: ", paste(left_available, collapse=", "), "\n")
cat("Right seeds found:", paste(right_available, collapse=", "), "\n")

# Compute average vector for each pole
left_vec  <- colMeans(emb[left_available, ])
right_vec <- colMeans(emb[right_available, ])

# The left-right axis is the difference
lr_axis <- right_vec - left_vec

# Project any word onto this axis
project_word <- function(word, axis, embeddings) {
  if (!(word %in% rownames(embeddings))) return(NA)
  cosine_sim(embeddings[word, ], axis)
}

# Test words: a mix of politically charged and neutral terms
test_words <- c(
  # Expected left-leaning
  "equality", "welfare", "poverty", "healthcare", "unions",
  "progressive", "redistribution",
  # Expected right-leaning
  "enterprise", "deregulation", "taxpayer", "freedom",
  "profit", "competition", "privatization",
  # Contested / neutral
  "immigration", "climate", "education", "security",
  "democracy", "economy", "family", "tradition",
  # Control: should be neutral
  "table", "bicycle", "sandwich", "umbrella"
)

# Filter to available words
test_words <- test_words[test_words %in% rownames(emb)]

# Compute projections
projections <- data.frame(
  word = test_words,
  lr_score = sapply(test_words, project_word, axis = lr_axis, embeddings = emb)
)
projections <- projections %>% arrange(lr_score)

# Print results
cat("\nLeft-Right projections (negative = left, positive = right):\n\n")
print(projections, row.names = FALSE)

# Visualize
projections$word <- factor(projections$word, levels = projections$word)

ggplot(projections, aes(x = lr_score, y = word,
                         fill = ifelse(lr_score > 0, "Right", "Left"))) +
  geom_col() +
  scale_fill_manual(values = c("Left" = "#E4003B", "Right" = "#0087DC")) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  labs(
    title = "Projecting Words onto a Left-Right Axis",
    subtitle = "Defined by seed words: solidarity/equality/welfare vs. market/enterprise/liberty",
    x = "Left ← → Right (cosine similarity with axis)",
    y = NULL,
    fill = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "top")

# Discussion:
# - Do the projections match your intuition?
# - Are the "neutral" control words actually neutral?
# - What happens if you change the seed words?
#   (Try replacing some seeds and rerunning — are results robust?)


# -----------------------------------------------------------------------------
# 9. DETECTING BIAS IN EMBEDDINGS (WEAT-style)
# -----------------------------------------------------------------------------
# Following Caliskan et al. (2017):
# Are male names closer to career words and female names closer to family words?

male_names   <- c("john", "paul", "james", "robert", "michael",
                   "david", "richard", "charles", "joseph", "thomas")
female_names <- c("mary", "patricia", "jennifer", "elizabeth", "linda",
                   "susan", "margaret", "dorothy", "sarah", "jessica")

career_words <- c("executive", "management", "professional", "corporation",
                   "salary", "office", "business", "career")
family_words <- c("home", "parents", "children", "family",
                   "cousins", "marriage", "wedding", "relatives")

# Filter to available words
male_avail   <- male_names[male_names %in% rownames(emb)]
female_avail <- female_names[female_names %in% rownames(emb)]
career_avail <- career_words[career_words %in% rownames(emb)]
family_avail <- family_words[family_words %in% rownames(emb)]

# Compute average similarity of male names to career vs. family
male_career_sim <- mean(sapply(male_avail, function(name)
  mean(sapply(career_avail, function(w) cosine_sim(emb[name,], emb[w,])))))

male_family_sim <- mean(sapply(male_avail, function(name)
  mean(sapply(family_avail, function(w) cosine_sim(emb[name,], emb[w,])))))

female_career_sim <- mean(sapply(female_avail, function(name)
  mean(sapply(career_avail, function(w) cosine_sim(emb[name,], emb[w,])))))

female_family_sim <- mean(sapply(female_avail, function(name)
  mean(sapply(family_avail, function(w) cosine_sim(emb[name,], emb[w,])))))

cat("\nAverage cosine similarity:\n")
cat("  Male names   — Career words:", round(male_career_sim, 4), "\n")
cat("  Male names   — Family words:", round(male_family_sim, 4), "\n")
cat("  Female names — Career words:", round(female_career_sim, 4), "\n")
cat("  Female names — Family words:", round(female_family_sim, 4), "\n")

cat("\nDifferences:\n")
cat("  Male (career - family):  ", round(male_career_sim - male_family_sim, 4), "\n")
cat("  Female (career - family):", round(female_career_sim - female_family_sim, 4), "\n")
cat("  Bias effect: male names are",
    round(male_career_sim - male_family_sim - (female_career_sim - female_family_sim), 4),
    "more career-associated than female names\n")

# Visualize
bias_df <- data.frame(
  Group = rep(c("Male names", "Female names"), each = 2),
  Association = rep(c("Career", "Family"), 2),
  Similarity = c(male_career_sim, male_family_sim,
                  female_career_sim, female_family_sim)
)

ggplot(bias_df, aes(x = Group, y = Similarity, fill = Association)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("Career" = "#1B2A4A", "Family" = "#D4763C")) +
  labs(
    title = "Gender-Career Bias in Word Embeddings",
    subtitle = "Replicating Caliskan et al. (2017) with GloVe vectors",
    y = "Average cosine similarity",
    x = NULL,
    fill = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "top")

# Discussion:
# - Is there a measurable gender-career bias?
# - Where does this bias come from? (The training corpus: Wikipedia + news)
# - What would happen if we trained embeddings on parliamentary speeches only?
# - What are the implications for using embeddings in downstream classifiers?


# =============================================================================
# DISCUSSION QUESTIONS
# =============================================================================
#
# 1. Look at the nearest neighbors for "immigration" (section 3).
#    What frame does the embedding reflect — economic, security, cultural?
#    What does this tell you about the training corpus?
#
# 2. In the analogy section (6), which analogies worked and which failed?
#    Can you find a political analogy that works? Why are political
#    analogies harder than geographic ones?
#
# 3. In the PCA plot (section 7), do the categories cluster as expected?
#    Which cross-category proximities are interesting?
#
# 4. Look at your left-right projections (section 8). Change two seed
#    words on each side and rerun. How much do the projections change?
#    What does this tell you about the robustness of the method?
#
# 5. In the bias analysis (section 9), try replacing career/family with
#    another attribute pair: science/arts, leadership/support, or
#    strength/sensitivity. Do you find other biases?
#
# =============================================================================
# HOMEWORK ASSIGNMENT
# =============================================================================
#
# Task: Use pre-trained embeddings to study a political concept of your choice.
#
# 1. Choose a politically interesting word or concept (e.g., populism,
#    sovereignty, austerity, multiculturalism, patriotism).
#
# 2. Find its 20 nearest neighbors. Inspect and discuss them.
#
# 3. Define a meaningful axis using seed words (e.g., left vs. right,
#    inclusive vs. exclusive, national vs. international). Project your
#    concept and 10+ related words onto this axis.
#
# 4. Visualize ~20 related words using PCA.
#
# 5. Test at least one bias dimension (gender, class, race, age)
#    using WEAT-style seed words.
#
# 6. Write a 1.5-page interpretation:
#    - What does the embedding tell you about how this concept is
#      positioned in the linguistic landscape?
#    - What surprised you?
#    - What are the limitations?
#
# Submit: R script + report. Due before Lecture 4.
# =============================================================================
