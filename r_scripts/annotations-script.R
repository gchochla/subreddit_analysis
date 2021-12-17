#######################################
# Annotations Coding
#######################################

if (!(require(pacman))) {
  install.packages(pacman)
}

#Load libraries

p_load(tidyverse, summarytools, magrittr,
       rmarkdown, knitr, kableExtra,
       scales, irr)

options(scipen = 100)

theme_set(theme_minimal())

setwd('C:/Users/asafm/Documents/GitHub/willpower-project/annotations')

#######################################
# Data Cleaning
#######################################

# 50-post annotations

lif_50 <- list.files(pattern = '.+50.+.csv')

anot_1 <- lif_50 %>% 
  map_dfr(~ read_csv(., col_types = cols())) %>%
  select(-(Willpower)) %>% 
  rename(sit_str = `Situational Strategy`,
         #sit_obst = `Situational obstacle`,
         willpower = `Effortful inhibition`) %>% 
  pivot_wider(names_from = Annotator, values_from = willpower:sit_str) %>%
  mutate(wp = rowMeans(select(., starts_with('willpower'))),
         sit_str = rowMeans(select(., starts_with('sit_str'))),
         sit_obst = NA,
         source = 'Calibration') %>% 
  select(post_id, wp, sit_obst, sit_str, source)

# 300-post annotations

lif_300 <- list.files(pattern = '.+300.+.csv')

anot_2 <- lif_300 %>% 
  map_dfr(~ read_csv(., col_types = cols())) %>% 
  select(-(Notes)) %>% 
  rename(sit_str = `Situational Strategy`,
         sit_obst = `Situational Obstacle`,
         wp = Willpower) %>%
  arrange(post_id, Annotator) %>% 
  group_by(post_id) %>% 
  mutate(Annotator = paste0(Annotator, '_ann', row_number())) %>%
  ungroup() %>% 
  separate(Annotator, c('ann_name', 'ann_id')) %>%
  pivot_wider(id_cols = 'post_id', 
              names_from = ann_id, values_from = wp:ann_name) %>% 
  rename_with(~ str_replace(., 'ann_', '')) %>% 
  mutate(diff_wp = wp_ann1 - wp_ann2,
         diff_obst = sit_obst_ann1 - sit_obst_ann2,
         diff_str = sit_str_ann1 - sit_str_ann2,
         wp = rowMeans(select(., starts_with('wp_'))),
         sit_obst = rowMeans(select(., starts_with('sit_obst_'))),
         sit_str = rowMeans(select(., starts_with('sit_str_'))),
         source = 'Main'
         ) %>% 
  select(post_id, wp, sit_obst, sit_str, source)

#write_csv(anot_2, 'complete-annotations-300.csv')

# Write both .csvs

write_csv(rbind(anot_1, anot_2), 'Annotated c25k data.csv')

# Number of rows with same ratings

sum(anot_2$diff_wp == anot_2$diff_obst &
      anot_2$diff_obst == anot_2$diff_str)

# Number of unanimous rows

sum(anot_2$diff_wp == 0 &
      anot_2$diff_obst == 0 &
      anot_2$diff_str == 0)

# % Unanimous

sum(anot_2$diff_wp == 0 &
      anot_2$diff_obst == 0 &
      anot_2$diff_str == 0) / nrow(anot_2)

# Inter-rater reliability: ICC

icc(select(anot_2, wp_ann1, wp_ann2), model = 'twoway')
icc(select(anot_2, sit_obst_ann1, sit_obst_ann2), model = 'twoway')
icc(select(anot_2, sit_str_ann1, sit_str_ann2), model = 'twoway')

# Seeing best performance based on rater and domain - Willpower

icc(anot_2 %>% filter(name_ann1 != 'Asaf' & name_ann2 != 'Asaf') %>% select(wp_ann1, wp_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Georgios' & name_ann2 != 'Georgios') %>% select(wp_ann1, wp_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Milad' & name_ann2 != 'Milad') %>% select(wp_ann1, wp_ann2),
    model = 'twoway')

# Seeing best performance based on rater and domain - Situational Obstacle

icc(anot_2 %>% filter(name_ann1 != 'Asaf' & name_ann2 != 'Asaf') %>% select(sit_obst_ann1, sit_obst_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Georgios' & name_ann2 != 'Georgios') %>% select(sit_obst_ann1, sit_obst_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Milad' & name_ann2 != 'Milad') %>% select(sit_obst_ann1, sit_obst_ann2),
    model = 'twoway')

# Seeing best performance based on rater and domain - Situational Strategy

icc(anot_2 %>% filter(name_ann1 != 'Asaf' & name_ann2 != 'Asaf') %>% select(sit_str_ann1, sit_str_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Georgios' & name_ann2 != 'Georgios') %>% select(sit_str_ann1, sit_str_ann2),
    model = 'twoway')

icc(anot_2 %>% filter(name_ann1 != 'Milad' & name_ann2 != 'Milad') %>% select(sit_str_ann1, sit_str_ann2),
    model = 'twoway')




# Simple correlation

cor <- correlation(select(anot, effort_Milad:last_col()))

cor %<>%
  filter(str_detect(Parameter1, 'effort') & str_detect(Parameter2, 'effort') |
         str_detect(Parameter1, 'uphold') & str_detect(Parameter2, 'uphold') |
         #str_detect(Parameter1, 'sit_obst') & str_detect(Parameter2, 'sit_obst') |
         str_detect(Parameter1, 'sit_str') & str_detect(Parameter2, 'sit_str')
           ) %>%
  mutate(across(c(r:t), ~ round(., digits = 2)),
         p = pvalue(p)) %>% 
  select(Parameter1, Parameter2, r, p)

#layers <- visualisation_recipe(summary(cor, redundant = T))
#layers
#plot(layers)

cor %>% 
  kable() %>% 
  kable_styling()

# ICC

icc(select(anot, effort_Milad, effort_Asaf, effort_Georgios), model = 'twoway')
icc(select(anot, uphold_Milad, uphold_Asaf, uphold_Georgios), model = 'twoway')
icc(select(anot, sit_str_Milad, sit_str_Asaf, sit_str_Georgios), model = 'twoway')
