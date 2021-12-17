#######################################
# Willpower (C25K / Loseit) Analyses
#######################################

if (!(require(pacman))) {
  install.packages(pacman)
}

set.seed(1)

#Load libraries

p_load(tidyverse, summarytools,
       magrittr, lubridate,
       quanteda, yarrr,
       sjPlot, performance,
       lme4, lmerTest,
       tidytext, topicmodels,
       rbenchmark, textclean)

options(scipen = 100)

theme_set(theme_minimal())

setwd('C:/Users/asafm/Documents/GitHub/willpower-project')

# Generic code for text cleaning

# mutate(text_clean = text %>% 
#          replace_non_ascii() %>% 
#          replace_html(symbol = F) %>% # remove html tag
#          str_replace_all("[0-9]", " ") %>% 
#          str_replace_all("[-|]", " ") %>% # replace "-" with space
#          tolower() %>% #lowercase
#          str_remove_all("coronavirus|covid 19|covid|canadian|canadians") %>%  # remove common words
#          replace_symbol() %>%
#          replace_contraction() %>% 
#          replace_word_elongation() %>%  # lengthen shortened word
#          str_replace_all("[[:punct:]]", " ") %>% # remove punctuation
#          str_replace_all(" dr ", " doctor ") %>% 
#          make_plural() %>%
#          str_replace_all(" s ", " ") %>%  
#          str_squish() %>% # remove double whitespace
#          str_trim() # remove whitespace at the start and end of the text
# )

# Helper function for pre-merge checks

pre_merge <- function(x, y) {
  return(tibble(
    xandy = length(union(x, y)),
    xnoty = length(setdiff(x, y)),
    ynotx = length(setdiff(y, x))
  ) %>% 
    pivot_longer(xandy:ynotx) %>% 
    mutate(value_com = scales::comma(value),
           prop = scales::percent(prop.table(value), accuracy = 1)
    )
  )
}

#######################################
# Data Cleaning
#######################################

# # Read in data
# 
# dat_raw <- read_csv('datasets/C25K/C25K-11835-pushshift.csv', na = 'N/A')
# 
# dat <- dat_raw %>%
#   mutate(timestamp = ymd_hms(timestamp)) %>%
#   unite(text, c(title, text), sep = ' ', na.rm = T) %>%
#   mutate(text = text %>% 
#            tolower() %>% 
#            replace_symbol() %>% 
#            replace_contraction() %>% 
#            replace_word_elongation() %>% 
#          str_squish() %>%
#            str_trim(),
#          timestamp_op = case_when(post_id == parent_post_id ~ timestamp)) %>%
#   group_by(author) %>%
#   mutate(day = ceiling(interval(min(timestamp_op, na.rm = T), timestamp+1) / days(1)), # Day since first OP - adding one second to avoid "day 0"
#          week = ceiling(interval(min(timestamp_op, na.rm = T), timestamp+1) / weeks(1)), # Week since first OP - adding one second to avoid "week 0"
#          op_nineweek = sum(day <= 63 & post_id == parent_post_id), # Number of original posts in first 9 weeks of program
#          activ_nineweek = sum(day <= 63), # Number of posts or comments in first 9 weeks of program
#          range_op = interval(min(timestamp_op), max(timestamp)) / days(1),
#          numweek = length(unique(week[week <= 9])),
#          succeed = ifelse(numweek >= 8, 1, 0),
#          succeed_lax = ifelse(activ_nineweek > 8, 1, 0),
#          numpost = n()) %>%
#   ungroup() %>%
#   mutate(text = str_replace(text, 'week.{0,4}(\\d).{0,4}day.{0,4}(\\d)', 'w\\1d\\2'),
#          text = str_replace(text, 'w.{0,4}(\\d).{0,4}d.{0,4}(\\d)', 'w\\1d\\2'),
#          text = str_replace(text, '(\\d).{0,4}day.{0,4}(\\d).{0,4}week', 'w\\2d\\1')) %>% # Standardize "W1D1" markers
#   mutate(text = map_chr(tokens(text, remove_punct = T), ~ paste(., collapse = ' ')),
#          numtok = str_count(text, '\\b\\w'),
#          thumbnail = ifelse(!(is.na(thumbnail)), 1, 0),
#          is_c25k = 1) %>%
#   select(-(c(upvote_ratio, url, timestamp_op)))
# 
# rm(dat_raw)
# 
# write_csv(dat, 'datasets/C25K/C25K-11835-pushshift-tokenized.csv')
# 
# ddr <- read_tsv('datasets/C25K/doc_loadings.tsv')
# 
# dat <- left_join(dat, ddr, by = c('post_id' = 'ID'))
# 
# write_csv(dat, 'datasets/C25K/C25K-11835-pushshift-tokenized.csv')
# 
# rm(ddr)
# 
# # Read in and process baseline data
# 
# base_raw <- read_csv('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-250-baseline.csv', na = 'N/A')
# 
# base_raw %<>%
#   group_by(author) %>%
#   slice_sample(n = 50)
# 
# Sys.time()
# 
# base <- base_raw %>%
#   mutate(timestamp = ymd_hms(timestamp)) %>%
#   unite(text, c(title, text), sep = ' ', na.rm = T) %>%
#   mutate(text = str_replace(text, ''', "'"),
#          text = str_replace(text, ''', "'"),
#          text = text %>%
#            tolower() %>%
#            replace_symbol() %>%
#            replace_contraction() %>%
#            replace_word_elongation() %>%
#            str_squish() %>%
#            str_trim()) %>%
#   group_by(author) %>%
#   mutate(day = round(interval(min(timestamp), timestamp+1) / days(1)), # Day since first post
#          range = interval(min(timestamp), max(timestamp)) / days(1),
#          numpost = n()) %>%
#   ungroup() %>%
#   mutate(text = map_chr(tokens(text, remove_punct = T), ~ paste(., collapse = ' ')),
#          numtok = str_count(text, '\\b\\w'),
#          is_c25k = 0) %>% # tokenize words and add c25k label
#   select(-url)
# 
# Sys.time()
# 
# rm(base_raw)
#
# ddr <- read_tsv('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/doc_loadings.tsv')
# 
# base <- left_join(base, ddr, by = c('post_id' = 'ID'))
# 
# write_csv(base, 'C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-50-baseline-tokenized.csv')
# 
# rm(ddr)
# 
# write_csv(base, 'C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-50-baseline-tokenized.csv')

dat <- read_csv('datasets/C25K/C25K-11835-pushshift-tokenized.csv')

dat %<>%
  select(-c(n_awards, wp_dict)) %>% 
  mutate(subreddit = 'c25k') %>% 
  select(post_id:author, subreddit, OP:wp_dict_big)

# Sanity check - all firstday posts are op

freq(filter(dat, day == 1)[['OP']])

base <- read_csv('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-50-baseline-tokenized.csv')

base %<>%
  select(-c(range, wp_dict)) %>% 
  mutate(OP = double(length = nrow(base)),
         thumbnail = double(length = nrow(base)),
         week = double(length = nrow(base)),
         op_nineweek = double(length = nrow(base)),
         activ_nineweek = double(length = nrow(base)),
         range_op = double(length = nrow(base)),
         succeed = double(length = nrow(base)),
         succeed_lax = double(length = nrow(base)),
         numweek = double(length = nrow(base)),
         is_fitness = ifelse(subreddit %in% c('loseit', 'intermittentfasting', 'progresspics', 
                                                     'xxfitness', 'CICO', '1200isplenty', 'keto',
                                                     'Fitness', 'fatlogic', 'couchto5k', 'Noom', 
                                                     'orangetheory', 'fasting', 'bodyweightfitness',
                                                     'cycling', 'fitbit', 'c25k', 'running', 'stopdrinking','WeightLossAdvice'), 1, 0),
         text = str_replace(text, 'self.control', 'self-control'))

freq(base$is_fitness) # Number of removed fitness related posts other than c25k

base %<>%
  filter(is_fitness == 0) %>% 
  select(post_id:subreddit, OP, text:score, thumbnail, timestamp:day, 
         week, op_nineweek, activ_nineweek, range_op, numweek, succeed, succeed_lax, numweek, numpost:wp_dict_big)  

#identical(names(dat), names(base))

names(dat)
names(base)

dat <- rbind(dat, base)

dat %<>%
  mutate(across(starts_with('wp_'), ~ as.numeric(.))) %>%
  rename(wp_dict = wp_dict_big) %>% 
  group_by(author, is_c25k) %>% 
  mutate(wp_avg = mean(wp_dict, na.rm = T),
         numtok_avg = mean(numtok, na.rm = T)) %>% 
  ungroup() %>% 
  mutate(wp_c = wp_dict - wp_avg,
         interc = 1,
         i_count = str_count(text, "i|i'"))

rm(base)

# Exclusions

nrow(dat)
freq(dat$is_c25k)
length(unique(dat$author))

paste('There were', length(dat$author[dat$author == '[deleted]']), 'posts from users with deleted accounts')
paste('There were', length(unique(dat$author[dat$numpost > 150 & dat$author != '[deleted]'])), 'users with over 150 posts')

# P's with over 150 posts

dat %<>%
  filter(numpost < 150)

nrow(dat)
length(unique(dat$author))

# Add anotated posts

ann_raw <- read_csv('annotations/complete-annotations-300.csv')

dat <- left_join(dat, ann_raw, by = 'post_id')

# Create between-person data

dat_bet <- dat %>%
  select(author, is_c25k, op_nineweek, activ_nineweek, succeed, succeed_lax, numweek, numpost,
         wp_avg, wp_avg_big) %>%
  group_by(author, is_c25k) %>% 
  slice_sample() %>% 
  pivot_longer(numpost:wp_avg_big) %>%
  mutate(is_c25k = ifelse(is_c25k == 1, 'c25k','base')) %>% 
  unite(name, c(is_c25k, name), sep = '_') %>% 
  pivot_wider(names_from = name, values_from = value) %>%
  group_by(author) %>% 
  summarize(across(op_nineweek:c25k_wp_avg_big, ~ sum(., na.rm = T))) %>% 
  mutate(interc = 1)

dat_25 <- filter(dat, subreddit == 'c25k' & week > -4)

ann <- dat_25 %>% 
  filter(!(is.na(wp_ann1)))

nrow(ann)

# # Correlation between DDR and annotations
# 
# ggplot(ann, aes(x = wp_ann1, y = wp_dict_big)) +
#   geom_jitter() +
#   geom_smooth(method = 'lm')
# 
# cor.test(ann$wp_ann1, ann$wp_dict_big)

# Create labeled DDR vectors

# doc_vec <- read_csv('datasets/C25K/doc_vectors.tsv')
# 
# nrow(doc_vec)
# 
# doc_vec <- left_join(doc_vec, select(dat, post_id, wp_dict, is_c25k, is_fitness), by = c('ID' = 'post_id'))
# 
# write_csv(doc_vec, 'doc_vectors_labeled.csv')
# 
# doc_vec_base <- read_csv('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/doc_vectors.tsv')
# 
# nrow(doc_vec_base)
# 
# doc_vec_base <- left_join(doc_vec_base, select(dat, post_id, wp_dict, is_c25k, is_fitness), by = c('ID' = 'post_id'))
# 
# write_csv(doc_vec_base, 'C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/baseline_doc_vectors_labeled.csv')


##################################
# Subset data for annotations
##################################

# wp_quant <- quantile(dat_25$wp_dict_big, probs = seq(0, 1, .2), na.rm = T)
# 
# dat_samp <- dat_25 %>%
#   filter(str_count(text, '\\s') > 1) %>% # Remove posts/comments with just 1-2 words
#   mutate(wp_dec = as.numeric(cut(.[['wp_dict_big']], breaks = wp_quant))) %>% 
#   filter(!(is.na(wp_dec))) %>%
#   group_by(wp_dec) %>% 
#   sample_n(60) %>% 
#   ungroup() %>% 
#   select(post_id, text) %>% 
#   mutate(Willpower = character(length = nrow(.)),
#          `Situational Obstacle` = character(length = nrow(.)),
#          `Situational Strategy` = character(length = nrow(.))) %>% 
#   sample_frac(., 1L)
# 
# write_csv(dat_samp[1:((2/3)*nrow(dat_samp)),], paste0('Subsample - ', nrow(dat_samp), ' posts for annotating - Georgios.csv'))
# write_csv(dat_samp[((1/3)*nrow(dat_samp) + 1):nrow(dat_samp),], paste0('Subsample - ', nrow(dat_samp), ' posts for annotating - Milad.csv'))
# write_csv(dat_samp[c(((2/3)*nrow(dat_samp) + 1):nrow(dat_samp),
#                      (1:((1/3)*nrow(dat_samp)))),], 
#           paste0('Subsample - ', nrow(dat_samp), ' posts for annotating - Asaf.csv'))



##################################
# Loseit Analyses
##################################

setwd('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/loseit')

lose_raw <- read_csv('loseit-70749-pushshift.csv', na = 'N/A')

nrow(lose_raw)

Sys.time()

lose <- lose_raw %>%
  dplyr::select(-(url)) %>% 
  mutate(timestamp = ymd_hms(timestamp)) %>%
  filter(author %in% sample(author, 10000)) %>% 
  group_by(author) %>%
  filter(n() < 500) %>% # Remove participants who posted in loseit more than 500 times
  ungroup() %>% 
  unite(text, c(title, text), sep = ' ', na.rm = T) %>%
  mutate(text = str_replace(text, ''', "'"),
         text = str_replace(text, ''', "'"),
         text = text %>%
           tolower() %>%
           replace_symbol() %>%
           replace_contraction() %>%
           replace_word_elongation() %>%
           str_squish() %>%
           str_trim()) %>%
  group_by(author) %>%
  mutate(day = round(interval(min(timestamp), timestamp+1) / days(1)), # Day since first post
         range = interval(min(timestamp), max(timestamp)) / days(1),
         numpost = n()) %>%
  ungroup() %>%
  mutate(text = map_chr(tokens(text, remove_punct = T), ~ paste(., collapse = ' ')),
         numtok = str_count(text, '\\b\\w')) # tokenize words and add c25k label
  
Sys.time()

nrow(lose)

rm(lose_raw)

# ddr <- read_tsv('C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/doc_loadings.tsv')
# 
# base <- left_join(base, ddr, by = c('post_id' = 'ID'))
# 
# write_csv(base, 'C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-50-baseline-tokenized.csv')
# 
# rm(ddr)
# 
# write_csv(base, 'C:/Users/asafm/Desktop/USC/Research/willpower-in-language/baseline_data/C25K-11835-pushshift-50-baseline-tokenized.csv')


#######################################
# Analyses
#######################################

#####################################
# Descriptive Analyses - author level
#####################################

view(freq(dat_bet$numweek))

pirateplot(numweek ~ interc, data = filter(dat_bet, numweek  <= 9),
           point.cex = .01, point.o = .01,
           bw = 0.5, 
           xlab = '', ylab = 'Number of weeks in C25K',
           cex.lab = 1.2, pal = '#5dC1ff')

pirateplot(numweek ~ interc, data = dat_bet, plot = F, inf.method = 'iqr')

ggplot(dat_bet, aes(x = numweek)) +
  geom_density(alpha = 0.7, fill = '#5dC1ff', bw = 0.4) +
  geom_vline(xintercept = quantile(dat_bet$numweek, probs = .9), size = 2, color = '#E0B0FF') +
  scale_x_continuous(limits = c(0, 10))

view(freq(dat_bet$succeed))
view(freq(dat_bet$succeed_lax, cumul = F, round.digits = 0))

#####################################
# Descriptive Analyses - post level
#####################################

view(freq(dat$score))

view(freq(dat$numtok))

#######################################
# Sanity check - Classifier
#######################################

quantile(dat$wp_dict, probs = seq(0, 1, .05), na.rm = T)
quantile(dat$wp_dict_big, probs = seq(0, 1, .05), na.rm = T)

dat_ex <- dat_25 %>% 
  mutate(hilo = ifelse(wp_dict > .27 | wp_dict_big > .49, 1,
                ifelse(wp_dict < .10 | wp_dict_big < .25, 0, NA))) %>% 
  filter(!(is.na(hilo))) %>% 
  select(post_id, parent_post_id, author, text, hilo, everything())

View(dat_ex %>% arrange(desc(wp_dict_big)))

ggplot(dat_bet, aes(x = c25k_wp_avg_big, y = op_nineweek)) +
  geom_jitter(color = 'dodgerblue', alpha = 0.7) +
  labs(x = 'Mean willpower loading',
       y = 'Original C25K posts in first nine weeks') +
  theme(text = element_text(size = 15))

# Model predicting weekly posting in first 9 weeks

summary(week_mod <- MASS::glm.nb(op_nineweek ~ c25k_numpost + c25k_wp_avg_big,
                          data = dat_bet))

car::vif(week_mod)

# Predicting subreddit

pirateplot(wp_dict_big ~ is_c25k, data = filter(dat, wp_dict_big > 0),
           point.cex = .001, point.o = .001,
           ylab = 'Willpower language', xlab = 'CouchTo5k',
           cex.lab = 1.3)

pirateplot(wp_dict_big ~ is_c25k, data = dat, plot = F)
pirateplot(wp_dict_big ~ is_c25k, data = dat, plot = F, avg.line.fun = median)

#dat <- filter(dat, !(fitness == 1 & is_c25k == 0))

pirateplot(wp_dict ~ is_c25k, data = dat, point.cex = .01, point.o = .01)

pirateplot(wp_dict ~ is_c25k, data = dat, plot = F)
pirateplot(wp_dict ~ is_c25k, data = dat, plot = F, avg.line.fun = median)

pirateplot(wp_dict ~ fitness, data = dat, plot = F)
pirateplot(wp_dict ~ fitness, data = dat, plot = F, avg.line.fun = median)

#######################################
# Exploratory word counting
#######################################

pirateplot(i_count ~ week, data = filter(dat_25, i_count < 200),
           point.cex = .01, point.o = .01)

pirateplot(i_count ~ week, data = dat_25, plot = F)

#######################################
# stLDA-C
#######################################

source('C:/Users/asafm/Documents/GitHub/stLDA-C_public/scripts/setup.R')
source('C:/Users/asafm/Documents/GitHub/stLDA-C_public/scripts/helper_functions.R')
source('C:/Users/asafm/Documents/GitHub/stLDA-C_public/scripts/gibbs_functions.R')

dat_tidy <- dat_25 %>% 
  filter(week <= 9) %>% 
  #slice_sample(n = 10000) %>% 
  select(post_id, text)

glimpse(dat_tidy)
  
dat_corp <- corpus(dat_tidy,
                   docid_field = 'post_id',
                   text_field = 'text')

dat_corp <- tokens(dat_corp,
                   remove_punct = T,
                   remove_url = T,
                   remove_numbers = T)

dat_dfm <- dfm(dat_corp) %>%
  dfm_remove(stopwords('english')) %>%
  dfm_wordstem() %>% 
  dfm_trim(min_termfreq = 0.95, termfreq_type = 'quantile', # Remove the 5% least frequent words (order is descending)
           max_docfreq = 0.2, docfreq_type = 'prop') # Include only words that appear in at most 10% of documents

dtm <- convert(dat_dfm, to = 'topicmodels')

# Sys.time()
# 
# lda <- LDA(dtm, k = 100, control = list(seed = 1))
# 
# Sys.time()

topics <- tidy(lda,matrix = "beta")

lda_top <- topics %>% 
  group_by(topic) %>% 
  slice_max(beta, n = 10) %>% 
  ungroup() %>% 
  arrange(topic, -beta) %>% 
  group_by(topic) %>%
  mutate(rank = row_number())

lda_top_wide <- lda_top %>% 
  select(-beta) %>% 
  pivot_wider(names_from = topic, values_from = term) %>% 
  write_csv('lda/lda-100-top-terms.csv')

topics_sp <- topics %>% 
  pivot_wider(names_from = 'topic', values_from = 'beta')

gam <- tidy(lda, matrix = 'gamma')

gam %<>% 
  pivot_wider(names_from = topic, values_from = gamma,
              names_prefix = 'gam_') %>% 
  write_csv('lda/lda-100-document-topic-loadings.csv')

#20-topic LDA

# Sys.time()
# 
# lda_20 <- LDA(dtm, k = 20, control = list(seed = 1))
# 
# Sys.time()
# 
# topics_20 <- tidy(lda_20,matrix = "beta")
# 
# lda_20_top <- topics %>% 
#   group_by(topic) %>% 
#   slice_max(beta, n = 5) %>% 
#   ungroup() %>% 
#   arrange(topic, -beta)

# Out the box code from stlda-c

nT <- lda@k

words <- topics_sp$term
tw_true <- topics_sp[,2:(nT+1)] %>% t

nC <- 100
nUC <- 10
nU <- nUC*nC
nW <- ncol(tw_true)

alpha_true <- matrix(1,nrow = nC,ncol = nT)

ca_true <- rep(1:nC,times = nUC) %>% sort
ut_true <- sapply(ca_true,function(c) rgamma(n = nT,shape = alpha_true[c,],rate = 1) %>% {./sum(.)}) %>% t

nDperU <- 40
users <- lapply(1:nU,function(u) rep(u,nDperU)) %>% unlist
ta_true <- lapply(users,function(u) sample(1:nT,size=1,prob = ut_true[u,])) %>% unlist

dw <- sapply(ta_true,function(t) rmultinom(n=1,size = 13,prob = tw_true[t,])) %>% t
ut_true_counts <- sapply(1:nU,function(u) sapply(1:nT,function(t) sum(ta_true == t & users ==u))) %>% t 

# stlda <- collapsed_gibbs_1topic_clusters(alpha = 1, eta = .1, nu = 1,
#                                          users = users,dw = dw,
#                                          nT = lda@k, nC = 10,
#                                          niter = 25, seed = 1,
#                                          mcmc_update = T, nClusterIter = 100,
#                                          mu_scale = 0, sigma_scale = 100,   
#                                          prop_scale_center = 100, alphag_sample_method = 'componentwise',
#                                          print_clusters = T)
# 
# # save.image(file = 'lda_stlda.Rdata')

# Merge data and LDA

pre_merge(dat$post_id, gam$document)

dat_lda <- left_join(dat, gam, by = c('post_id' = 'document'))

write_csv(dat_lda, 'lda/full-c52k-with-lda-100-loadings.csv')

# library(randomForest)
# 
# rf_lda <- randomForest(wp_dict ~ ., 
#                        data = dat_lda %>%
#                          select(wp_dict, starts_with('gam')) %>%
#                          filter(!(is.na(wp_dict)))
#                          )

cor_lda <- correlation::correlation(data = select(dat_lda, wp_dict),
                                    data2 = select(dat_lda, starts_with('gam')),
                                    method = 'spearman')

write_csv(cor_lda %>% 
            select(-c(S:n_Obs, CI)) %>% 
            arrange(rho) %>% 
            mutate(across(c(rho, starts_with('CI')),
                          ~ round(., digits = 2))),
          'lda/correlation-lda-ddr.csv')

lda_ann <- filter(dat_lda, !(is.na(wp_ann1)), !(is.na(gam_1)))

cor_lda_ann <- correlation::correlation(data = select(lda_ann, wp_dict),
                                        data2 = select(lda_ann, starts_with('gam')),
                                        method = 'spearman')

write_csv(cor_lda_ann %>% 
            select(-c(S:n_Obs, CI)) %>% 
            arrange(rho) %>% 
            mutate(across(c(rho, starts_with('CI')),
                          ~ round(., digits = 2))), 
          'lda/correlation-lda-annot.csv')


#######################################
# Structural Topic modelling
#######################################

p_load(stm)




#######################################
# Inferential Analyses
#######################################

# Database of quasi-successes (4 or more weeks of posting/commenting in first 9 weeks)

dat_suc <- filter(dat_25, numweek >= 4)

nrow(dat_suc)
length(unique(dat_suc$author))

pirateplot(wp_dict_big ~ week, data = filter(dat_suc, 
                                         week <= 9 & week >= -4 &
                                           wp_dict_big < .6 & wp_dict > .25),
           point.cex = .01, point.o = .01)

summary(fitmod <- glmer(is_c25k ~ numtok +  wp_dict_big + (1|author), data = dat,
                        family = binomial))

tab_model(fitmod, standardized = T,
          pred.labels = c('Intercept','Word count', 'Willpower language'),
          dv.labels = NULL)

sjPlot::plot_model(fitmod, type = 'pred', terms = 'wp_dict_big')

performance(fitmod)

#############################
# Predicting success
#############################

summary(suc_mod <- glm(succeed_lax ~ c25k_wp_avg_big + base_wp_avg_big, data = dat_bet,
                        family = binomial))

car::vif(suc_mod)

plot_model(suc_mod, type = 'pred', terms = c('c25k_wp_avg_big','base_wp_avg_big'))

#######################################
# Social reinforcement analyses
#######################################

ggplot(dat_25, aes(x = wp_dict_big, y = score)) +
  geom_jitter(color = 'dodgerblue', alpha = 0.7) +
  scale_y_continuous(limits = c(0, 500)) +
  labs(x = 'Willpower language', y = 'Upvote - Downvote') +
  theme(text = element_text(size = 15))

summary(scor_mod <- glmer.nb(score ~ numtok + wp_dict_big + (1|author), data = filter(dat_25, score >= 0)))

plot_model(scor_mod, type = 'pred', terms = c('wp_dict_big')) +
  labs(x = 'Willpower language', y = 'Score', title = '') +
  theme(text = element_text(size = 15))

# Comment analyses

dat_25_op <- dat_25 %>% 
  group_by(parent_post_id) %>% 
  mutate(num_com = n()) %>% 
  filter(parent_post_id == post_id)

ggplot(dat_25_op, aes(x = wp_dict_big, y = num_com)) +
  geom_jitter(color = 'dodgerblue', alpha = 0.7) +
  labs(x = 'Willpower language', y = 'Number of comments on post') +
  theme(text = element_text(size = 15))

hist(dat_25_op$num_com)

summary(com_mod <- glmer.nb(num_com ~ numtok + wp_dict_big + (1|author), data = dat_25_op))

summary(com_mod <- MASS::glm.nb(num_com ~ numtok + wp_dict_big, data = dat_25_op))

#######################################
# Topic modeling
#######################################

p_load(topicmodels, tidytext, quanteda, rbenchmark)

# Change "slice sample" to match desired sample size

dat_tidy <- dat_25 %>%
  filter(week <= 9) %>%
  slice_sample(n = 1000) %>%
  unnest_tokens(word, text, token = 'ptb')

data(stop_words)

dat_tidy %<>%
  filter(!(word %in% stop_words$word),
         !(str_detect(word, '^\\d*$')))

dat_count <- dat_tidy %>%
  group_by(post_id) %>%
  count(word, sort = T)

#dat_tf_idf <- dat_count %>% 
#  bind_tf_idf(word, post_id, n)

dat_dtm <- dat_count %>%
  cast_dtm(post_id, word, n)

rm(dat)
rm(dat_bet)
rm(dat_tidy)
rm(dat_count)
rm(stop_words)

# Uncomment to run LDA

#benchmark(dat_lda <- LDA(dat_dtm, k = 10, control = list(seed = 1), method = 'gibbs'))

lda_topics <- tidy(dat_lda, matrix = 'beta')

lda_top_terms <- lda_topics %>% 
  group_by(topic) %>% 
  slice_max(beta, n = 10) %>% 
  ungroup() %>% 
  arrange(topic, -beta)

# lda_top_terms %>%
#   mutate(term = reorder_within(term, beta, topic)) %>%
#   ggplot(aes(beta, term, fill = factor(topic))) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~ topic, scales = "free") +
#   scale_y_reordered()


#######################################
# Network analyses
#######################################

p_load(igraph, tidygraph, ggraph)
import::from(sna, symmetrize)

dat_net <- dat %>% 
  group_by(parent_post_id) %>% 
  mutate(parent_author = author[OP = max(OP)]) %>%
  filter(author != parent_author) %>%
  group_by(author, parent_author) %>%
  summarize(weight = n())
  
dat_att <- dat_net

dat_net %<>%
  select(author, parent_author, weight)

grafdat <- graph_from_data_frame(d = dat_net)

grafdat <- as_tbl_graph(grafdat)

# number of unique nodes

length(unique(gvec(grafdat, author)))

View(grafdat)

plot(grafdat)

# Analyze components of network

comp <- components(grafdat, mode = 'weak')
count_components(grafdat)
comp

sum(comp$csize > 5)
max(comp$csize)

which(comp$csize==max(comp$csize))
comp$membership[comp$membership==1]


# To edit - network metrics

# Density

edge_density(grafdat)

reciprocity(grafdat)

transitivity(grafdat, type = 'global')

dyad_census(grafdat)

# Metrics by misinformation or not

indeg <- degree(grafdat, mode = 'in', normalized = F)
outdeg <- degree(grafdat, mode = 'out', normalized = F)
betw <- betweenness(grafdat)
betw_std <- betweenness(grafdat, normalized = T)
inclose <- closeness(grafdat, mode = 'in')
inclose_std <- closeness(grafdat, mode = 'in', normalized = T)
outclose <- closeness(grafdat, mode = 'out')
outclose_std <- closeness(grafdat, mode = 'out', normalized = T)

# Add metrics as node attributes

grafdat <- grafdat %>% 
  mutate(indeg = indeg,
         inclose = inclose,
         betw = betw)

# Centrality

(indeg_centr <- centralize(indeg, 
                           theoretical.max = centr_degree_tmax(grafdat, mode = 'in'), normalized = T))
(outdeg_centr <- centralize(outdeg, 
                            theoretical.max = centr_degree_tmax(grafdat, mode = 'out'), normalized = T))
(between_centr <- centralize(betw, 
                             theoretical.max = centr_betw_tmax(grafdat), normalized = T))
(inclose_centr <- centralize(inclose,
                             theoretical.max = centr_clo_tmax(grafdat, mode = 'in'), normalized = T))
(outclose_centr <- centralize(outclose,
                              theoretical.max = centr_clo_tmax(grafdat, mode = 'out'), normalized = T))

# Clusters

un_grafdat <- as_tbl_graph(as.undirected(grafdat, mode = 'collapse'))

un_grafdat <- un_grafdat %>% 
  mutate(
    indeg = degree(un_grafdat, mode = 'in', normalized = F),
    inclose = closeness(un_grafdat, mode = 'in', normalized = F),
    betw = betweenness(un_grafdat)
  )

cfg <- cluster_fast_greedy(un_grafdat)

modularity(cfg)

# Try to create adjacency matrix without the computer crashing

adj_mat <- as.matrix(as_adj(grafdat))

# Diameter and average path length

diameter(grafdat)
mean_distance(grafdat)

# 
# mean(indeg[tib$misinf_bin == 1], na.rm = T)
# mean(indeg[tib$misinf_bin == 0], na.rm = T)
# 
# mean(outdeg[tib$misinf_bin == 1], na.rm = T)
# mean(outdeg[tib$misinf_bin == 0], na.rm = T)
# 
# mean(betw[tib$misinf_bin == 1], na.rm = T)
# mean(betw[tib$misinf_bin == 0], na.rm = T)
# mean(betw_std[tib$misinf_bin == 1], na.rm = T)
# mean(betw_std[tib$misinf_bin == 0], na.rm = T)
# 
# mean(inclose[tib$misinf_bin == 1], na.rm = T)
# mean(inclose[tib$misinf_bin == 0], na.rm = T)
# mean(inclose_std[tib$misinf_bin == 1], na.rm = T)
# mean(inclose_std[tib$misinf_bin == 0], na.rm = T)
# 
# mean(outclose[tib$misinf_bin == 1], na.rm = T)
# mean(outclose[tib$misinf_bin == 0], na.rm = T)
# mean(outclose_std[tib$misinf_bin == 1], na.rm = T)
# mean(outclose_std[tib$misinf_bin == 0], na.rm = T)

# Transitivity

triad_census(grafdat)



