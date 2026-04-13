# =============================================================================
# Lecture 1 — Text as Data: Foundations & Preprocessing
# Practical Session: Exploring U.K. House of Commons Speeches
#
# Methods Seminar: Multimodal Computational Methods in Political Science
# Computational Social Science Program, LMU Munich
# =============================================================================
#
# Before running this script:
#   1. Install required packages (see section 0 below, run once)
#   2. Download the ParlSpeech V2 dataset from:
#      https://doi.org/10.7910/DVN/L4OAKN
#   3. Extract the UK House of Commons file: Corp_HouseOfCommons_V2.rds
#   4. Place it in the same folder as this script (or adjust the path below)
# =============================================================================


# -----------------------------------------------------------------------------
# 0. SETUP: Install and load packages
# -----------------------------------------------------------------------------

# Run this block ONCE to install packages (comment out after first run)
# install.packages(c(
#   "quanteda",
#   "quanteda.textstats",
#   "quanteda.textplots",
#   "readtext",
#   "tidyverse",
#   "ggplot2"
# ))

library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(tidyverse)



# -----------------------------------------------------------------------------
# 1. LOAD THE DATA
# -----------------------------------------------------------------------------
"C:/Users/my27n/OneDrive/Рабочий стол/Humboldt_Fellowship/LLM_course/Lecture#1/Corp_HouseOfCommons_V2.rds"
# Load the ParlSpeech V2 UK House of Commons file
# (change the path to match where you saved the file)
speeches_raw <- readRDS("C:/Users/my27n/OneDrive/Рабочий стол/Humboldt_Fellowship/LLM_course/Lecture#1/Corp_HouseOfCommons_V2.rds")

# Inspect the structure
glimpse(speeches_raw)

# The ParlSpeech V2 format includes these key columns:
#   text          - the speech text
#   speaker       - speaker name
#   party         - party affiliation
#   date          - date of the speech
#   agenda        - topic of the debate
#   parliament    - parliament number
#   terms         - number of words

# Filter to a manageable subset: one parliament, substantive speeches only
# The 2017-2019 parliament (57th) is a good choice — Brexit debates included
speeches <- speeches_raw %>%
  filter(date >= "2017-06-08", date <= "2019-11-06") %>%
  filter(terms >= 50) %>%                      # drop very short speeches
  filter(!is.na(party), party != "")            # drop missing party info

# Check how many speeches we have and the party distribution
nrow(speeches)
table(speeches$party) #Lab - Labour, Con - Conservative, LibDem - Liberal Democrats, SNP - Scottish National Party, GPEW - Green Party of England and Wales

# Keep only the main parties (drop tiny parties with too few speeches)
main_parties <- c("Con", "Lab", "LibDem",
                  "SNP", "GPEW")
speeches <- speeches %>%
  filter(party %in% main_parties)

# Optional: take a random sample for faster in-class work
# (comment out if you want to use the full corpus)
set.seed(42)
speeches <- speeches %>% slice_sample(n = 5000)

nrow(speeches)
table(speeches$party)


# -----------------------------------------------------------------------------
# 2. CREATE A QUANTEDA CORPUS
# -----------------------------------------------------------------------------

# A quanteda corpus stores texts alongside their metadata (docvars)
corp <- corpus(speeches, text_field = "text")

# Inspect
summary(corp, n = 5)

# Look at the first speech (first 500 characters)
cat(substr(as.character(corp[1]), 1, 500))

# Access metadata
head(docvars(corp))


# -----------------------------------------------------------------------------
# 3. PREPROCESSING: TOKENIZE, CLEAN, LOWERCASE
# -----------------------------------------------------------------------------

# Step 1: Tokenize
# - remove_punct: drop commas, periods, etc.
# - remove_numbers: drop numeric tokens
# - remove_url: drop URLs (rare in parliamentary text, but good practice)
# - remove_symbols: drop $, %, etc.
toks <- tokens(corp,
               remove_punct   = TRUE,
               remove_numbers = TRUE,
               remove_url     = TRUE,
               remove_symbols = TRUE)

# Inspect: first 30 tokens of the first document
head(toks[[1]], 30)

# Step 2: Lowercase everything
toks <- tokens_tolower(toks)

# Step 3: Remove English stop words
# Note: stopwords("en") includes "not", "no", "against" etc.
# For sentiment analysis you'd want to keep those, but for this task it's fine.
toks <- tokens_remove(toks, stopwords("en"))

# Step 4: Also remove parliament-specific filler words
parliament_stopwords <- c("hon", "friend", "gentleman", "lady", "member",
                          "members", "house", "speaker", "minister",
                          "right", "government", "secretary", "state")
toks <- tokens_remove(toks, parliament_stopwords)

# Inspect the result
head(toks[[1]], 30)
# Optional: generate bigrams for richer features
# toks_bi <- tokens_ngrams(toks, n = 1:2)


# -----------------------------------------------------------------------------
# 4. BUILD THE DOCUMENT-FEATURE MATRIX (DFM)
# -----------------------------------------------------------------------------

# Build the DFM (what we called DTM in the lecture — quanteda calls it DFM)
dfm_speeches <- dfm(toks)

# Basic statistics
ndoc(dfm_speeches) #documents, N speeches
nfeat(dfm_speeches) #N features
round(sparsity(dfm_speeches) * 100, 2) #Sparsity, that is share os zeroes in DFM

# Top 20 most frequent words in the corpus
topfeatures(dfm_speeches, 20)

# Trim rare and ubiquitous words
# - min_docfreq = 10: word must appear in at least 10 speeches
# - max_docfreq = 0.5 (proportion): remove words in more than 50% of speeches
dfm_trimmed <- dfm_trim(dfm_speeches,
                        min_docfreq = 10 / ndoc(dfm_speeches), #here we drop words that are very rarre (appear in less then 10 documents). We can not indicate 10 docs straight away and that is why we should express it in proportion 10/number of documents total
                        max_docfreq = 0.5, #here we drop words that are very common (appear in more that 50% of documents)
                        docfreq_type = "prop")


nfeat(dfm_trimmed)#N features after trimming 


# -----------------------------------------------------------------------------
# 5. EXPLORE: TOP WORDS BY PARTY
# -----------------------------------------------------------------------------

# Group the DFM by party
dfm_party <- dfm_group(dfm_trimmed, groups = party)

# Top words for each party (raw frequency)
topfeatures(dfm_party["Con", ], 15)#for Conservative

topfeatures(dfm_party["Lab", ], 15)#Labour

topfeatures(dfm_party["LibDem", ], 15)#Liberal Democrats

topfeatures(dfm_party["SNP", ], 15) #Scotish National Party

topfeatures(dfm_party["GPEW", ], 15) #Greens


# -----------------------------------------------------------------------------
# 6. TF-IDF WEIGHTING
# -----------------------------------------------------------------------------

# Compute TF-IDF weights
dfm_tfidf_party <- dfm_tfidf(dfm_party)

topfeatures(dfm_tfidf_party["Con", ], 15)
topfeatures(dfm_tfidf_party["Lab", ], 15)
topfeatures(dfm_tfidf_party["LibDem", ], 15)
topfeatures(dfm_tfidf_party["SNP", ], 15)
topfeatures(dfm_tfidf_party["GPEW", ], 15)
# Compare with raw counts above — TF-IDF should surface more distinctive words


# -----------------------------------------------------------------------------
# 7. KEYNESS: STATISTICALLY DISTINCTIVE WORDS PER PARTY
# -----------------------------------------------------------------------------

# Keyness compares the frequency of words in a target group vs. reference
# using a chi-squared test. It's more principled than just looking at top TF-IDF.

# Example: what words distinguish Labour from all other parties?
# We group the DFM into two: Labour vs. non-Labour
dfm_lab_vs_rest <- dfm_trimmed %>%
  dfm_group(groups = ifelse(docvars(dfm_trimmed, "party") == "Lab",
                            "Lab", "Other"))

keyness_labour <- textstat_keyness(dfm_lab_vs_rest, target = "Lab")

# Plot the results
textplot_keyness(keyness_labour, n = 30,
                 color = c("#E4003B", "gray60")) +
  ggtitle("Words distinctive of Labour vs. all other parties")

# Same for Conservative
dfm_con_vs_rest <- dfm_trimmed %>%
  dfm_group(groups = ifelse(docvars(dfm_trimmed, "party") == "Con",
                            "Con", "Other"))

keyness_cons <- textstat_keyness(dfm_con_vs_rest, target = "Con")

textplot_keyness(keyness_cons, n = 30,
                 color = c("#0087DC", "gray60")) +
  ggtitle("Words distinctive of Conservative vs. all other parties")


# -----------------------------------------------------------------------------
# 8. VISUALIZATION: TOP WORDS BY PARTY (GGPLOT)
# -----------------------------------------------------------------------------

# Group speeches by party
dfm_party <- dfm_group(dfm_speeches, groups = docvars(dfm_speeches, "party"))

# Compute TF-IDF on the grouped dfm
dfm_party_tfidf <- dfm_tfidf(dfm_party)

# Get top 10 terms within each party document
freq_by_party <- textstat_frequency(
  dfm_party_tfidf,
  n = 10,
  groups = docnames(dfm_party_tfidf),
  force = TRUE
)

# Clean up the data for plotting
freq_by_party <- freq_by_party %>%
  mutate(feature = tidytext::reorder_within(feature, frequency, group))

# UK party colors
uk_party_colors <- c(
  "Con" = "#0087DC",
  "Lab" = "#E4003B",
  "LibDem" = "#FAA61A",
  "SNP" = "#FDF38E",
  "GPEW" = "#6AB023"
)

ggplot(freq_by_party,
       aes(x = feature, y = frequency, fill = group)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ group, scales = "free_y") +
  tidytext::scale_x_reordered() +
  scale_fill_manual(values = uk_party_colors) +
  labs(
    title = "Top 10 TF-IDF Words by Party",
    subtitle = "UK House of Commons, 2017–2019",
    x = NULL,
    y = "TF-IDF score"
  ) +
  theme(strip.text = element_text(face = "bold", size = 11))

# -----------------------------------------------------------------------------
# 9. WORD CLOUD (OPTIONAL, FOR FUN)
# -----------------------------------------------------------------------------

# Simple comparison word cloud across parties
# (shows which words are distinctive of each group)


dfm_party <- dfm_group(dfm_speeches, groups = docvars(dfm_speeches, "party"))
keep_parties <- c("Con", "Lab", "LibDem", "GPEW", "SNP")# keep only main parties 
dfm_party <- dfm_party[docnames(dfm_party) %in% keep_parties, ]
dfm_party_trim <- dfm_trim(dfm_party, min_termfreq = 2)# trim low-frequency words on the count dfm

party_colors <- c(
  "Con" = "#0087DC",
  "Lab" = "#E4003B",
  "LibDem" = "#FAA61A",
  "GPEW" = "#6AB023",
  "SNP" = "#FDF38E"
)

party_order <- docnames(dfm_party_trim)

textplot_wordcloud(
  dfm_party_trim,
  comparison = TRUE,
  max_words = 100,
  color = party_colors[docnames(dfm_party_trim)]
)

legend(
  "bottomleft",
  legend = names(party_colors),
  text.col = party_colors,
  bty = "n"
)

# -----------------------------------------------------------------------------
# 10. SENSITIVITY CHECK: DOES PREPROCESSING MATTER?
# -----------------------------------------------------------------------------

# Let's run an alternative pipeline WITHOUT stop word removal
# and see if the top words change substantially

toks_alt <- tokens(corp,
                   remove_punct   = TRUE,
                   remove_numbers = TRUE,
                   remove_url     = TRUE) %>%
  tokens_tolower()
# Note: we skip tokens_remove(stopwords("en"))

dfm_alt <- dfm(toks_alt) %>%
  dfm_trim(min_docfreq = 10/ndoc(toks_alt), max_docfreq = 0.5, docfreq_type = "prop")

topfeatures(dfm_alt, 15)#Top 15 words WITHOUT stopword removal


topfeatures(dfm_trimmed, 15) #Top 15 words WITH stopword removal (our main pipeline)

# Discuss: How different are these? What do we lose without stop word removal?


# =============================================================================
# DISCUSSION QUESTIONS FOR THE PRACTICAL (reflect at home)
# =============================================================================
#
# 1. Look at the top 15 raw frequency words per party (section 5).
#    Are they what you expected? Any surprises?
#
# 2. Compare raw frequency (section 5) with TF-IDF (section 6) for the same
#    party. Which gives you more substantively meaningful "party words"?
#    Why?
#
# 3. In the keyness plot (section 7), find one word you didn't expect
#    to distinguish Labour from other parties. What might explain it?
#
# 4. Look at the parliament-specific stopwords we removed in section 3.
#    Try re-running the pipeline without removing them. What changes?
#    What does this teach you about domain-specific preprocessing?
#
# 5. Try filtering to only speeches from 2019 (the Brexit endgame).
#    How do the top words change compared to the full 2017-2019 period?
#
# =============================================================================
# HOMEWORK ASSIGNMENT (NOT COMPULSORY)
# =============================================================================
#
# Task: Apply the preprocessing pipeline to a political text corpus of your
# choice.
#
# 1. Choose a corpus: another parliament from ParlSpeech V2
#
# 2. Preprocess with explicit justification for each choice: tokenization,
#    stop words, lowercasing, trimming thresholds.
#
# 3. Build a DFM. Report: number of documents, vocabulary size, sparsity.
#
# 4. Compute TF-IDF. Identify the 10 most distinctive words per group
#    (party, time period, whatever grouping is relevant for your corpus).
#
# 5. Sensitivity check: rerun with one different preprocessing choice and
#    report whether the top-10 words change.
#
# 6. Write a 1-page interpretation of the vocabulary differences.
#
# Submit: R script + 1-page write-up. Due before Lecture 2.
# =============================================================================