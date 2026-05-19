# =============================================================================
# Lecture 4 — Document Representations & Topic Models
# Practical Session: Structural Topic Model on UK House of Commons Speeches
#
# Methods Seminar: Multimodal Computational Methods in Political Science
# Computational Social Science Program, LMU Munich
# =============================================================================
#
# In this practical we will:
#   1. Build document-level vectors by averaging word embeddings (warm-up)
#   2. Preprocess the UK Commons corpus for topic modeling
#   3. Fit a Structural Topic Model (STM) with K = 20 topics
#   4. Interpret topics using top-probability and FREX words
#   5. Label topics manually (the researcher's job!)
#   6. Estimate covariate effects: topic prevalence by party and over time
#   7. Choose K using diagnostics (searchK)
#   8. Validate the model (word intrusion, representative documents)
#
# Prerequisites:
#   - ParlSpeech V2 UK Commons file (Corp_HouseOfCommons_V2.rds) — see Lecture 1
#   - Packages installed (see section 0)
# =============================================================================


# -----------------------------------------------------------------------------
# 0. SETUP
# -----------------------------------------------------------------------------
rm(list=ls(all=TRUE))
# Run ONCE to install (comment out after first run)
# install.packages(c(
#   "quanteda",
#   "quanteda.textstats",
#   "stm",
#   "stminsights",   # optional: interactive Shiny explorer
#   "tidyverse",
#   "ggplot2"
# ))


library(quanteda)
library(quanteda.textstats)
library(stm)
library(tidyverse)

set.seed(42)  # Topic models are stochastic — always set a seed!
theme_set(theme_minimal(base_size = 12))


# -----------------------------------------------------------------------------
# 1. WARM-UP: DOCUMENT VECTORS BY AVERAGING WORD EMBEDDINGS
# -----------------------------------------------------------------------------
#
# Before topic models, a quick demonstration of the simplest document
# representation: average the word embeddings of all words in a document.
# (This connects Lecture 3 to Lecture 4.)

# Load the UK Commons corpus
speeches_raw <- readRDS("C:/Users/my27n/OneDrive/Рабочий стол/Humboldt_Fellowship/LLM_course/Lecture#1/Corp_HouseOfCommons_V2.rds")


speeches <- speeches_raw %>%
  filter(date >= "2017-06-08", date <= "2019-11-06") %>%
  filter(terms >= 50) %>%
  filter(!is.na(party), party != "") %>%
  filter(party %in% c("Con", "Lab", "LibDem", "SNP"))   #Conservative (Con), Labour(lab), Liberal Democrat (LibDem), SNP (Scottish National Party)

# Stratified sample: equal number per party
per_party <- 1000 # manageable size for in-class work
speeches <- speeches %>%
  group_by(party) %>%
  slice_sample(n = per_party) %>%
  ungroup()


cat("Loaded", nrow(speeches), "speeches\n")
print(table(speeches$party))

rm(speeches_raw)

# --- Optional embedding demo (requires GloVe from Lecture 3) ---
# If you have the GloVe matrix `emb` from Lecture 3 in memory, you can
# build averaged document vectors like this:
#
# build_doc_vector <- function(text, embeddings) {
#   words <- unlist(strsplit(tolower(text), "\\s+"))
#   words <- words[words %in% rownames(embeddings)]
#   if (length(words) == 0) return(rep(0, ncol(embeddings)))
#   colMeans(embeddings[words, , drop = FALSE])
# }
# doc_vecs <- t(sapply(speeches$text[1:100], build_doc_vector, embeddings = emb))
# dim(doc_vecs)   # 100 documents x 300 dimensions
#
# These dense document vectors could feed into the classifiers from Lecture 2.
# Today, however, we focus on the unsupervised approach: topic models.


# -----------------------------------------------------------------------------
# 2. PREPROCESSING FOR TOPIC MODELS
# -----------------------------------------------------------------------------
#
# Topic models need slightly different preprocessing than classification:
#   - Stemming is acceptable (we care about themes, not exact word forms)
#   - Aggressive trimming helps (removes noise, speeds up estimation)
#   - Removing procedural/filler words is important for parliamentary text

corp <- corpus(speeches, text_field = "text")

toks <- tokens(corp,
               remove_punct   = TRUE,
               remove_numbers = TRUE,
               remove_url     = TRUE,
               remove_symbols = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords("en")) %>%
  tokens_remove(c(
    # Parliamentary procedural / filler words — these dominate otherwise
    "hon", "friend", "gentleman", "lady", "member", "members",
    "house", "speaker", "right", "honourable", "minister",
    "government", "secretary", "state", "deputy", "mr", "mrs",
    "ms", "sir", "madam", "today", "will", "can", "must", "may",
    "make", "made", "also", "would", "could", "should", "one",
    "two", "many", "much", "well", "way", "give", "given", "take"
  )) %>%
  tokens_wordstem(language = "en")   # stem — fine for topic models

#Compare (just for learning purposes) the original text vs. Preprocessed 
speeches$text[1] #original
toks[["text1"]] #<preprocessed 

#Drop very rare and very common words (in essence somehow similar to the logic of tf-idf)
dfm_speeches <- dfm(toks) %>%
  dfm_trim(min_docfreq = 15,   docfreq_type = "count") %>% # word in >= 15 documents
  dfm_trim(max_docfreq = 0.40, docfreq_type = "prop")  # word in <= 40% of documents

cat("DFM:", ndoc(dfm_speeches), "documents x",
    nfeat(dfm_speeches), "features\n")

# Remove any documents that became empty after trimming
dfm_speeches <- dfm_subset(dfm_speeches, ntoken(dfm_speeches) > 0)
ndoc(dfm_speeches) #After removing empty docs


# -----------------------------------------------------------------------------
# 3. CONVERT TO STM FORMAT
# -----------------------------------------------------------------------------
#
# STM needs a specific input format. quanteda provides a converter.

stm_input <- convert(dfm_speeches, to = "stm")

# The converted object has three parts:
str(stm_input, max.level = 1)
# $documents : list, one entry per document (word indices + counts)
# $vocab     : character vector of unique words
# $meta      : data frame of document metadata (party, date, etc.)

# Prepare metadata: make sure types are correct
stm_input$meta$party <- as.factor(stm_input$meta$party)
stm_input$meta$year  <- as.numeric(format(
  as.Date(stm_input$meta$date), "%Y")) #to make it less noisy keep only year 

# Quick check
head(stm_input$meta[, c("party", "year")])
table(stm_input$meta$party)


# -----------------------------------------------------------------------------
# 4. FIT THE STRUCTURAL TOPIC MODEL
# -----------------------------------------------------------------------------
#
# Key arguments:
#   K          = number of topics (we start with 20)
#   prevalence = formula for what affects topic PREVALENCE
#                here: party (categorical) + smooth function of year
#   init.type  = "Spectral" gives deterministic, reproducible results

cat("\nFitting STM with K = 20 (this takes a few minutes)...\n")

stm_fit <- stm(
  documents  = stm_input$documents,
  vocab      = stm_input$vocab,
  K          = 20,
  prevalence = ~ party + year,
  data       = stm_input$meta,
  init.type  = "Spectral",
  max.em.its = 75,
  verbose    = TRUE
)

#-----------------------------Some intermediate explanation-----------#
#First, STM needs a starting guess for the topics. Spectral initialization builds this guess deterministically 
#(no randomness — same input always gives the same start, which is why it's reproducible).

#Gram matrix: a word-by-word co-occurrence matrix — how often each pair of words appears together across documents.
#Anchor words: the algorithm looks for words that are highly distinctive of a single topic ("anchors"). 
#Recovering initialization: using those anchor words, it reconstructs an initial estimate of all 20 topics.


#Spectral is preferred over the older random init: it lands on a sensible starting point
#instead of a random one, so you don't need to fit the model multiple times and pick the best run.

#Then STM fits via the EM algorithm, which alternates two steps until the model stabilizes:
# E-step (Expectation): holding the current topics fixed, estimate which topics each document is made of.
# M-step (Maximization): holding those document-topic assignments fixed, re-estimate the topics themselves (the word distributions).


#The per-word bound is a measure of model fit (technically a lower bound on the log-likelihood, normalized per word). 
#Two things to watch:
#If it's increasing (getting less negative): −7.084 → −6.931 → −6.887 → −6.868 → −6.858. Good — the model is improving each iteration.
#The relative change is shrinking fast? E.g.: 2.2% → 0.64% → 0.27% → 0.15%. This is the key diagnostic. 
#It shows that the model is converging — each iteration is buying you less and less improvement. 
#When the relative change drops below STM's tolerance (default ~1e-5), it stops automatically; otherwise it stops at max.em.its = 75
#---------------------------------------------------------------------#

# Save the model so you don't have to refit it
saveRDS(stm_fit, "stm_fit_K20.rds")
# Later: stm_fit <- readRDS("stm_fit_K20.rds")


# -----------------------------------------------------------------------------
# 5. INSPECT TOPICS
# -----------------------------------------------------------------------------

# Top words per topic, using FOUR different metrics:
#   Highest Prob: most frequent words in the topic
#   FREX:         frequent AND exclusive (best for interpretation!)
#   Lift:         words that appear disproportionately in this topic
#   Score:        another exclusivity-weighted measure
labelTopics(stm_fit, n = 10)

# Plot expected topic proportions across the corpus
plot(stm_fit, type = "summary", n = 6,
     main = "Topic Prevalence in UK Commons (2017-2019)")

# Find the most representative documents for selected topics
# (read these to understand what each topic actually captures)
thoughts_t1 <- findThoughts(stm_fit,
                            texts = speeches$text[as.integer(
                              rownames(stm_input$meta))],
                            topics = 1, n = 2)
plotQuote(thoughts_t1$docs[[1]], width = 60,
          main = "Representative documents: Topic 1")


# -----------------------------------------------------------------------------
# 6. LABEL TOPICS MANUALLY (THE RESEARCHER'S JOB)
# -----------------------------------------------------------------------------
#
# Look at labelTopics() output above. For EACH topic, read the
# Highest Prob and FREX words, then assign a substantive label.
# This requires YOUR political knowledge — the model only gives words.
#
# Fill in based on what YOU see (example labels shown — yours will differ):

topic_labels <- c(
  "T1: Road Safety & Transport",
  "T2: Parliamentary procedure",
  "T3: Agriculture",
  "T4: Education & Schools",
  "T5: Business & investment",
  "T6: Welfare & tax",
  "T7: Tributes",
  "T8: Brexit referendum",
  "T9: Justice & Policing",
  "T10: Scottish local constituency",
  "T11: Climate & immigration", #this is one is mixed 
  "T12: Middle East",
  "T13: Consumer protection",
  "T14: Local railway services",
  "T15: Procedural (not usefull)",#junk
  "T16: Legislation",
  "T17: Brexit: trade & negotiations",
  "T18: Health & NHS",
  "T19: Devolution: Scotland, NI & Wales",
  "T20: Women, pensions & workplace equality"
)
# NOTE: Be honest about "junk" topics (procedural language, boilerplate).
# They are real outputs and should be acknowledged, not hidden.

names(topic_labels) <- paste0("Topic", 1:20)


# -----------------------------------------------------------------------------
# 7. ESTIMATE COVARIATE EFFECTS
# -----------------------------------------------------------------------------
#
# This is what makes STM powerful for political science:
# we can ask how topic PREVALENCE varies by party and over time,
# with proper statistical uncertainty.

prep <- estimateEffect(
  formula  = 1:20 ~ party + year,
  stmobj   = stm_fit,
  metadata = stm_input$meta,
  uncertainty = "Global"
)

# --- Effect of PARTY on a chosen topic ---
# Example: which party emphasizes the Brexit topic most?
brexit_topic_procedural <- 8
brexit_topic_trade <- 17 

plot(prep,
     covariate = "party",
     topics    = brexit_topic_trade,
     model     = stm_fit,
     method    = "pointestimate",
     main      = paste("Prevalence of", topic_labels[brexit_topic_trade],
                       "by Party"),
     xlab      = "Expected topic proportion")
#"Constructive ambiguity" under Corbyn — deliberately avoiding firm positions on the single market/customs union to hold together
#a voter coalition split between Leave and Remain. Talking process lets you attack without committing; talking substance forces you to pick a side. 
#The model has, in effect, detected strategic ambiguity in the text.


# --- Effect of TIME on a topic ---
# Example: did the Brexit topic grow over 2017-2019? Compare with Education (should be stable over time)
education <- 4
plot(prep,
     covariate = "year",
     topics    = education,
     model     = stm_fit,
     method    = "continuous",
     main      = paste(topic_labels[education], "over time"),
     xlab      = "Year")

# --- Compare two parties on a topic directly ---

plot(prep,
     covariate  = "party",
     topics     = brexit_topic_trade,
     model      = stm_fit,
     method     = "difference",
     cov.value1 = "Con",
     cov.value2 = "Lab",
     xlim       = c(-0.01, 0.03),     # <-- force zero onto the plot
     main = "Difference in T17 (Brexit trade): Con - Lab",
     xlab = "<-- More Labour      More Conservative -->")

abline(v = 0, lty = 2, col = "red")

# -----------------------------------------------------------------------------
# 8. TOPIC CORRELATIONS
# -----------------------------------------------------------------------------
#
# Which topics tend to co-occur in the same speeches?

topic_corr <- topicCorr(stm_fit)
plot(topic_corr,
     vlabels = paste0("T", 1:20),
     main = "Topic Correlation Network")

#The topic correlation network shows which themes co-occur within speeches. 
#Read the edges, not the positions: connected clusters are coherent discourses, isolated nodes are siloed debates, 
#and a topic wired to everything is usually either rhetorical filler or a sign that topic needs splitting.

# -----------------------------------------------------------------------------
# 9. CHOOSING K: DIAGNOSTICS
# -----------------------------------------------------------------------------
#
# How do we know K = 20 was a good choice? Fit several values of K
# and compare diagnostics. WARNING: this is slow. 

K_candidates <- c(10, 20, 25)#in a real life example try various options (both greater and smaller than the current K)

search_results <- searchK(
  documents  = stm_input$documents,
  vocab      = stm_input$vocab,
  K          = K_candidates,
  prevalence = ~ party + s(year),
  data       = stm_input$meta,
  init.type  = "Spectral",
  N          = floor(0.1 * length(stm_input$documents))  # held-out sample
)

# Plot the diagnostics
plot(search_results)
# What to look for:
#   - Held-out likelihood: higher is better
#   - Semantic coherence:  higher is better (tends to fall as K grows)
#   - Exclusivity:         higher is better (tends to rise as K grows)
#   - Residuals:           lower is better
# The "best" K balances coherence and exclusivity AND is interpretable.

# A useful summary plot: coherence vs. exclusivity
search_df <- search_results$results %>%
  mutate(
    K      = as.integer(unlist(K)),
    semcoh = as.numeric(unlist(semcoh)),
    exclus = as.numeric(unlist(exclus))
  )

ggplot(search_df, aes(x = semcoh, y = exclus, label = K)) +
  geom_point(size = 4, color = "#1B2A4A") +
  geom_text(vjust = -1, color = "#D4763C", fontface = "bold") +
  labs(title = "Choosing K: Semantic Coherence vs. Exclusivity",
       subtitle = "Each point is a value of K. Top-right is better.",
       x = "Semantic Coherence",
       y = "Exclusivity")

# -----------------------------------------------------------------------------
# 10. VALIDATION
# -----------------------------------------------------------------------------

# --- (a) Face validity: can you label every topic? ---
# Already done in section 6. Flag any topic you couldn't label.

# --- (b) Semantic coherence & exclusivity per topic ---
topic_quality <- data.frame(
  topic      = 1:20,
  label      = topic_labels,
  coherence  = semanticCoherence(stm_fit, stm_input$documents),
  exclusivity = exclusivity(stm_fit)
)
print(topic_quality)

#Semantic coherence measures whether a topic's top words actually co-occur in the same documents. It is always negative; closer to zero = more coherent
#Exclusivity measures whether a topic's top words are distinctive to it rather than shared across many topics. Higher = more exclusive.

#Semantic coherence asks "do these words really co-occur?"; exclusivity asks "are these words distinctive to this topic?" 
#Both matter, but neither asks "is this topic about something useful" — that question only a human reading the speeches can answer.


# Plot: which topics are high-quality, which are weak?
ggplot(topic_quality, aes(x = coherence, y = exclusivity,
                          label = paste0("T", topic))) +
  geom_point(size = 3, color = "#1B2A4A") +
  geom_text(vjust = -1, size = 3, color = "#D4763C") +
  labs(title = "Per-Topic Quality",
       subtitle = "Topics in the lower-left may be junk or poorly defined",
       x = "Semantic Coherence", y = "Exclusivity")


# --- (c) Word intrusion test (do this with a partner!) ---
# For a given topic, show 5 real top words + 1 random intruder word.
# Can your partner spot the intruder? Good topics make this easy.
word_intrusion <- function(stm_model, topic, n_real = 5) {
  top_words <- labelTopics(stm_model, topics = topic,
                           n = n_real)$prob[1, ]
  # Pick a random word that is NOT a top word of this topic
  all_vocab <- stm_model$vocab
  intruder  <- sample(setdiff(all_vocab, top_words), 1)
  test_set  <- sample(c(top_words, intruder))
  cat("Topic", topic, "— which word does NOT belong?\n")
  cat(paste(seq_along(test_set), test_set, sep = ". "), sep = "\n")
  cat("\n(Intruder was:", intruder, ")\n\n")
}

# Run for a few topics
word_intrusion(stm_fit, topic = 1)
word_intrusion(stm_fit, topic = 7)
word_intrusion(stm_fit, topic = 15)   # try this on a "junk" topic


# -----------------------------------------------------------------------------
# 11. PUTTING IT TOGETHER: A SUBSTANTIVE FINDING
# -----------------------------------------------------------------------------
#
# Example research statement you could now make:
#
# "Fitting a 20-topic STM to UK House of Commons speeches (2017-2019),
#  we find that the Brexit/EU Withdrawal topic was the single most
#  prevalent theme, accounting for X% of expected topic proportion.
#  Its prevalence increased significantly over time (2017 -> 2019).
#  The Immigration topic was significantly more prevalent in
#  Conservative speeches than Labour speeches (difference = Y,
#  95% CI [..]), while the NHS topic showed the opposite pattern."
#
# All of these claims come WITH uncertainty estimates, directly
# from estimateEffect() — no separate regression needed.

# Overall topic prevalence table
topic_prevalence <- data.frame(
  label = topic_labels,
  prevalence = colMeans(stm_fit$theta)
) %>% arrange(desc(prevalence))

print(topic_prevalence)

ggplot(topic_prevalence,
       aes(x = reorder(label, prevalence), y = prevalence)) +
  geom_col(fill = "#1B2A4A") +
  coord_flip() +
  labs(title = "Topic Prevalence in UK Commons (2017-2019)",
       x = NULL, y = "Mean expected proportion per document")


# =============================================================================
# DISCUSSION QUESTIONS
# =============================================================================
#
# 1. FACE VALIDITY: Look at labelTopics() output. Could you label all 20
#    topics? Which were clear? Which looked like junk (procedural language)?
#    How many "real" substantive topics did you find?
#
# 2. FREX vs. HIGHEST PROB: For 2-3 topics, compare the "Highest Prob"
#    and "FREX" word lists. Which is more useful for interpretation? Why?
#
# 3. COVARIATE EFFECTS: Pick a topic where you EXPECT a party difference
#    (e.g., immigration, climate, NHS). Does estimateEffect() confirm your
#    expectation? Is the difference statistically distinguishable from zero?
#
# 4. TIME TRENDS: Did the Brexit topic grow over 2017-2019 as expected?
#    What does the shape of the time trend tell you?
#
# 5. CHOOSING K: Looking at the searchK diagnostics, was K = 20 a
#    reasonable choice? What would you choose and why? What changes
#    substantively if you refit with a very different K (e.g., K = 40)?
#
# 6. ROBUSTNESS: Rerun the STM with a different seed. Do the main
#    topics survive? Which topics are stable and which disappear?
#
# 7. WHAT DID THE MODEL DISCOVER? Which topics did the model find that
#    you would NOT have pre-specified if you had to list topics in
#    advance? This is the value of unsupervised discovery.
#
# =============================================================================
# HOMEWORK ASSIGNMENT
# =============================================================================
#
# Task: Fit and interpret a topic model for a political corpus.
#
# 1. Choose a corpus (UK Commons, another ParlSpeech parliament, EU
#    Parliament, party manifestos, newspaper editorials). Min. 1000 docs.
#
# 2. Preprocess appropriately. Document and justify your choices,
#    especially your stopword and trimming decisions.
#
# 3. Fit an STM with at least TWO values of K (e.g., K = 15 and K = 30).
#    Include at least one covariate (party, time, country, etc.).
#
# 4. For your chosen K, label ALL topics. Honestly flag junk topics.
#
# 5. Report ONE substantive finding using estimateEffect(), WITH its
#    uncertainty estimate (e.g., "topic X is more prevalent in party
#    A than party B, difference = ..., 95% CI = ...").
#
# 6. VALIDATE: run a word intrusion test on 3-5 topics. Report how
#    often the intruder was correctly identified.
#
# 7. Write a 2-page report: corpus, preprocessing, topic labels,
#    main finding, validation results, limitations.
#
# Submit: R script + 2-page report. Due before Lecture 5.
# =============================================================================