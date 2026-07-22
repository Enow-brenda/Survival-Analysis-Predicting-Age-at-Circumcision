# Loading important packages for the project
library(haven)
library(dplyr)
library(tidyr)
library(ggplot2)
library(survey)
library(survival)
library(survminer)
library(gtsummary)
library(janitor)
library(labelled)
library(tableone)
library(broom)
library(car)

# Set working directory to project folder

setwd("D:/Masters stuff/statistique/TP2-Survival Analysis")

getwd() # Check current working directory

##############################################
#1# loading dataset and identifying variables
##############################################

mr_data <- read_sav("dataset/CMMR71FL.SAV")

dim(mr_data) # Display dimensions

head(mr_data) # Display first rows

names(mr_data) # Variable names

required_vars <- c(
  "MV483",   # Circumcision status
  "MV483A",  # Age at circumcision
  "MV012",   # Current age
  "MV025",   # Residence
  "MV106",   # Education
  "MV190",   # Wealth index
  "MV501",   # Marital status
  "MV130",   # Religion
  "MV024",   # Region
  "MV159",   # Television exposure
  "MV158",   # Radio exposure
  "MV781",   # HIV testing
  "MV005",   # Sample weight
  "MV021",   # PSU
  "MV022"    # Strata
)

required_vars %in% names(mr_data)

# Select variables needed for analysis

df <- mr_data %>%
  select(
    MV483,
    MV483A,
    MV012,
    MV025,
    MV106,
    MV190,
    MV501,
    MV130,
    MV024,
    MV159,
    MV158,
    MV781,
    MV005,
    MV021,
    MV022
  )

# renaming variables

df <- df %>%
  rename(
    circumcision_status = MV483,
    age_circumcision    = MV483A,
    current_age         = MV012,
    residence           = MV025,
    education           = MV106,
    wealth              = MV190,
    marital_status      = MV501,
    religion            = MV130,
    region              = MV024,
    tv_exposure         = MV159,
    radio_exposure      = MV158,
    hiv_testing         = MV781,
    weight              = MV005,
    psu                 = MV021,
    strata              = MV022
  )

# checking new dataset

# Dimensions
dim(df)

# Variable types
str(df)

# display first rows
head(df)

# television exposure labels
attr(mr_data$MV159, "labels")

# radio exposure labels
attr(mr_data$MV158, "labels")

# circumcision status distribution
table(df$circumcision_status, useNA = "ifany")

# television exposure distribution
table(df$tv_exposure, useNA = "ifany")

# radio exposure distribution
table(df$radio_exposure, useNA = "ifany")

# HIV testing distribution
table(df$hiv_testing, useNA = "ifany")

# checking for missing values
colSums(is.na(df))

# checking for duplicates
sum(duplicated(df))

# inspecting duplicates
df[duplicated(df), ]

#########################################################
#2# Data Cleaning and Survival Variable Construction
#########################################################

#--------------------------------------------------------
# A. Remove "Don't know" circumcision status (code 8)
#--------------------------------------------------------

initial_sample <- nrow(df)

df <- df %>%
  filter(circumcision_status != 8)

removed_unknown_status <- initial_sample - nrow(df)

nrow(df)

#--------------------------------------------------------
# B. Convert unknown age at circumcision (98) to NA
#    (retain respondents in analysis)
#--------------------------------------------------------

df <- df %>%
  mutate(
    age_circumcision =
      ifelse(age_circumcision == 98,
             NA,
             age_circumcision)
  )

sum(is.na(df$age_circumcision))

#--------------------------------------------------------
# C. Recode DHS code 95
#    "Circumcised before age 5" -> age 2 years
#--------------------------------------------------------

df <- df %>%
  mutate(
    age_circumcision =
      ifelse(age_circumcision == 95,
             2,
             age_circumcision)
  )

table(df$age_circumcision, useNA = "ifany")

#--------------------------------------------------------
# D. Create event indicator
#    1 = circumcised
#    0 = not circumcised
#--------------------------------------------------------

df <- df %>%
  mutate(
    event = ifelse(circumcision_status == 1, 1, 0)
  )

table(df$event)

#--------------------------------------------------------
# E. Create survival time
#    Circumcised    -> age at circumcision
#    Uncircumcised  -> current age
#--------------------------------------------------------

df <- df %>%
  mutate(
    time = ifelse(
      event == 1,
      age_circumcision,
      current_age
    )
  )

summary(df$time)

#--------------------------------------------------------
# F. Create age groups
#--------------------------------------------------------

df <- df %>%
  mutate(
    age_group = case_when(
      current_age >= 15 & current_age <= 24 ~ "15-24",
      current_age >= 25 & current_age <= 34 ~ "25-34",
      current_age >= 35 & current_age <= 44 ~ "35-44",
      current_age >= 45 & current_age <= 54 ~ "45-54",
      current_age >= 55 ~ "55+"
    )
  )

table(df$age_group)

#--------------------------------------------------------
# G. Create media exposure variables
#--------------------------------------------------------

df <- df %>%
  mutate(
    tv_exposure = as_factor(tv_exposure),
    radio_exposure = as_factor(radio_exposure)
  )

table(df$tv_exposure)
table(df$radio_exposure)

#--------------------------------------------------------
# H. HIV testing status
#--------------------------------------------------------

df <- df %>%
  mutate(
    hiv_testing = as_factor(hiv_testing)
  )

table(df$hiv_testing)

#--------------------------------------------------------
# I. Create normalized survey weights
#--------------------------------------------------------

df <- df %>%
  mutate(
    weight_norm = weight / 1000000
  )

summary(df$weight_norm)

#--------------------------------------------------------
# J. Convert categorical variables to factors
#--------------------------------------------------------

df <- df %>%
  mutate(
    circumcision_status = factor(
      circumcision_status,
      levels = c(0,1),
      labels = c("No","Yes")
    ),
    age_group = factor(age_group),
    residence = as_factor(residence),
    education = as_factor(education),
    wealth = as_factor(wealth),
    marital_status = as_factor(marital_status),
    religion = as_factor(religion),
    region = as_factor(region)
  )

#--------------------------------------------------------
# K. Final dataset structure
#--------------------------------------------------------

dim(df)

str(df)

#--------------------------------------------------------
# L. Final sample summary
#--------------------------------------------------------

cat("Initial sample:", initial_sample, "\n")
cat("Removed unknown circumcision status (8):",
    removed_unknown_status, "\n")
cat("Unknown age at circumcision recoded to NA:",
    sum(is.na(df$age_circumcision)), "\n")
cat("Final analytical sample:", nrow(df), "\n")

# Circumcision distribution
table(df$circumcision_status)

# Event distribution
table(df$event)

# Age-group distribution
table(df$age_group)

#####################################################
#3# Creating the DHS Survey Design Object
#####################################################

#----------------------------------------------------
# A. Check survey variables
#----------------------------------------------------

summary(df$weight_norm)

table(df$strata)

length(unique(df$psu))

#----------------------------------------------------
# B. Handle lonely PSU strata
#----------------------------------------------------

options(survey.lonely.psu = "adjust")

#----------------------------------------------------
# C. Create DHS survey design object
#----------------------------------------------------

dhs_design <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight_norm,
  data = df,
  nest = TRUE
)

#----------------------------------------------------
# D. Inspect survey design
#----------------------------------------------------

dhs_design

summary(weights(dhs_design))

# Number of PSUs
length(unique(df$psu))

# Number of strata
length(unique(df$strata))

# Sample size
nrow(df)

#----------------------------------------------------
# E. Weighted circumcision prevalence
#----------------------------------------------------

svymean(
  ~factor(circumcision_status),
  design = dhs_design,
  na.rm = TRUE
)

#----------------------------------------------------
# F. Weighted HIV testing prevalence
#----------------------------------------------------

svymean(
  ~factor(hiv_testing),
  design = dhs_design,
  na.rm = TRUE
)

#----------------------------------------------------
# G. Weighted residence distribution
#----------------------------------------------------

svymean(
  ~factor(residence),
  design = dhs_design,
  na.rm = TRUE
)

#----------------------------------------------------
# H. Survey design object ready for analysis
#----------------------------------------------------

cat("Survey design object successfully created.\n")


##############################################
# 4. DESCRIPTIVE ANALYSIS
##############################################

# Age group
svytable(~age_group, dhs_design)
prop.table(svytable(~age_group, dhs_design)) * 100

# Residence
svytable(~residence, dhs_design)
prop.table(svytable(~residence, dhs_design)) * 100

# Education
svytable(~education, dhs_design)
prop.table(svytable(~education, dhs_design)) * 100

# Wealth
svytable(~wealth, dhs_design)
prop.table(svytable(~wealth, dhs_design)) * 100

# Marital status
svytable(~marital_status, dhs_design)
prop.table(svytable(~marital_status, dhs_design)) * 100

# Religion
svytable(~religion, dhs_design)
prop.table(svytable(~religion, dhs_design)) * 100

# Region
svytable(~region, dhs_design)
prop.table(svytable(~region, dhs_design)) * 100

# TV exposure
svytable(~tv_exposure, dhs_design)
prop.table(svytable(~tv_exposure, dhs_design)) * 100

# Radio exposure
svytable(~radio_exposure, dhs_design)
prop.table(svytable(~radio_exposure, dhs_design)) * 100

# HIV testing
svytable(~hiv_testing, dhs_design)
prop.table(svytable(~hiv_testing, dhs_design)) * 100

# Circumcision status
svytable(~circumcision_status, dhs_design)
prop.table(svytable(~circumcision_status, dhs_design)) * 100

table1 <- tbl_svysummary(
  dhs_design,
  by = circumcision_status,
  include = c(
    age_group,
    residence,
    education,
    wealth,
    marital_status,
    religion,
    region,
    tv_exposure,
    radio_exposure,
    hiv_testing
  ),
  percent = "column"
)

table1

####Comment###############
#Weighted descriptive statistics were calculated using DHS sampling weights to obtain nationally
#representative estimates. Most respondents were aged 15–24 years (38.1%), resided in urban areas (55.2%),
#and had secondary education (51.3%). Approximately 57.6% had ever been tested for HIV. 
#The weighted prevalence of male circumcision was 93.0%.

##############################################
# 5. DESCRIBING AGE AT CIRCUMCISION
##############################################

df %>%
  filter(event == 1) %>%
  summarise(
    mean = mean(age_circumcision, na.rm = TRUE),
    median = median(age_circumcision, na.rm = TRUE),
    sd = sd(age_circumcision, na.rm = TRUE),
    min = min(age_circumcision, na.rm = TRUE),
    max = max(age_circumcision, na.rm = TRUE)
  )

#####commment#######
#Among circumcised men, the mean age at circumcision was 4.8 years (SD = 4.16),
#while the median age was 3 years. The age at circumcision ranged from 0 to 40 years.
#The difference between the mean and median suggests a right-skewed distribution, 
#indicating that most circumcisions occurred during early childhood, with a smaller proportion occurring at older ages.

########################################################
#7# KAPLAN–MEIER (DHS WEIGHTED)
# Probability of remaining uncircumcised as age increases
########################################################

#-------------------------------------------------------
# Create analysis dataset (remove missing survival times)
#-------------------------------------------------------

df_surv <- df %>%
  filter(!is.na(time),
         !is.na(event))

#survey design
dhs_design_surv <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight_norm,
  data = df_surv,
  nest = TRUE
)

#-------------------------------------------------------
# Create survival object
#-------------------------------------------------------

surv_obj <- Surv(
  time = df_surv$time,
  event = df_surv$event
)

#-------------------------------------------------------
# Overall DHS-weighted Kaplan-Meier curve
#-------------------------------------------------------

km_fit <- svykm(
  surv_obj ~ 1,
  design = dhs_design_surv
)

km_fit

plot(
  km_fit,
  xlab = "Age (years)",
  ylab = "Probability of remaining uncircumcised",
  main = "DHS-Weighted Kaplan-Meier Survival Curve"
)

# The DHS-weighted Kaplan–Meier survival curve showed that male circumcision predominantly occurs during childhood.
# The estimated median age at circumcision was 4 years,with 25% of males circumcised by age 2 years and 75%
# circumcised by age 9 years. The sharp decline in the survival curve during early childhood indicates a high
# rate of circumcision at young ages, while the subsequent
# plateau suggests that relatively few circumcisions occur during adolescence and adulthood.

############################################################################
#8# STRATIFIED KAPLAN–MEIER CURVES
#We are checking whether the timing of circumcision differs between groups.
############################################################################

#a# residence(urban vs rural)
km_residence <- svykm(
  Surv(time, event) ~ residence,
  design = dhs_design_surv
)

fit_residence <- survfit(
  Surv(time, event) ~ residence,
  data = df_surv
)

ggsurvplot(
  fit_residence,
  data = df_surv,
  risk.table = FALSE,
  conf.int = FALSE,
  palette = "Dark2",
  legend.title = "Residence",
  xlab = "Age",
  ylab = "Probability of remaining uncircumcised",
  title = "Kaplan-Meier Curve by Residence"
)

#b# education
km_education <- svykm(
  Surv(time, event) ~ education,
  design = dhs_design_surv
)

fit_education <- survfit(
  Surv(time, event) ~ education,
  data = df_surv
)

ggsurvplot(
  fit_education,
  data = df_surv,
  risk.table = FALSE,
  conf.int = FALSE,
  palette = "Dark2",
  legend.title = "Education",
  xlab = "Age",
  ylab = "Probability of remaining uncircumcised",
  title = "Kaplan-Meier Curve by Education"
)

#c#wealth
km_wealth <- svykm(
  Surv(time, event) ~ wealth,
  design = dhs_design_surv
)

fit_wealth <- survfit(
  Surv(time, event) ~ wealth,
  data = df_surv
)

ggsurvplot(
  fit_wealth,
  data = df_surv,
  risk.table = FALSE,
  conf.int = FALSE,
  palette = "Dark2",
  legend.title = "Wealth Index",
  xlab = "Age",
  ylab = "Probability of remaining uncircumcised",
  title = "Kaplan-Meier Curve by Wealth Index"
)

#d#religion
km_religion <- svykm(
  Surv(time, event) ~ religion,
  design = dhs_design_surv
)

fit_religion <- survfit(
  Surv(time, event) ~ religion,
  data = df_surv
)

ggsurvplot(
  fit_religion,
  data = df_surv,
  risk.table = FALSE,
  conf.int = FALSE,
  palette = "Dark2",
  legend.title = "Religion",
  xlab = "Age",
  ylab = "Probability of remaining uncircumcised",
  title = "Kaplan-Meier Curve by Religion"
)

#e#region
km_region <- svykm(
  Surv(time, event) ~ region,
  design = dhs_design_surv
)

fit_region <- survfit(
  Surv(time, event) ~ region,
  data = df_surv
)

ggsurvplot(
  fit_region,
  data = df_surv,
  risk.table = FALSE,
  conf.int = FALSE,
  palette = "Dark2",
  legend.title = "Region",
  xlab = "Age",
  ylab = "Probability of remaining uncircumcised",
  title = "Kaplan-Meier Curve by Region"
)

svykm(surv_obj ~ residence, dhs_design_surv)
svykm(surv_obj ~ education, dhs_design_surv)
svykm(surv_obj ~ wealth, dhs_design_surv)
svykm(surv_obj ~ religion, dhs_design_surv)
svykm(surv_obj ~ region, dhs_design_surv)

########################################################
# Final quality checks before Cox regression
########################################################

# Event distribution
table(df_surv$event)

# Survival time summary
summary(df_surv$time)

# Missing values
sum(is.na(df_surv$time))
sum(is.na(df_surv$event))

# Range of survival times
range(df_surv$time)

# Factor levels
lapply(
  df_surv[, c(
    "age_group",
    "residence",
    "education",
    "wealth",
    "marital_status",
    "religion",
    "region",
    "tv_exposure",
    "radio_exposure",
    "hiv_testing"
  )],
  levels
)

# Final KM verification
svykm(surv_obj ~ 1, dhs_design_surv)

###comment####
#Survival times ranged from birth to 64 years, with a median age of 4 years and a mean age of 6.4 years.
#The higher mean relative to the median suggests a right-skewed distribution, 
#reflecting a small proportion of circumcisions occurring later in life.

###########################################################################
#8# LOG-RANK TESTS
# Testing whether circumcision timing differs across groups
###########################################################################

#--------------------------------------------------
# a) Age group
#--------------------------------------------------

logrank_age <- survdiff(
  surv_obj ~ age_group,
  data = df_surv
)

p_age <- 1 - pchisq(
  logrank_age$chisq,
  df = length(logrank_age$n) - 1
)

logrank_age
p_age


#--------------------------------------------------
# b) Residence
#--------------------------------------------------

logrank_residence <- survdiff(
  surv_obj ~ residence,
  data = df_surv
)

p_residence <- 1 - pchisq(
  logrank_residence$chisq,
  df = length(logrank_residence$n) - 1
)

logrank_residence
p_residence


#--------------------------------------------------
# c) Education
#--------------------------------------------------

logrank_education <- survdiff(
  surv_obj ~ education,
  data = df_surv
)

p_education <- 1 - pchisq(
  logrank_education$chisq,
  df = length(logrank_education$n) - 1
)

logrank_education
p_education


#--------------------------------------------------
# d) Wealth
#--------------------------------------------------

logrank_wealth <- survdiff(
  surv_obj ~ wealth,
  data = df_surv
)

p_wealth <- 1 - pchisq(
  logrank_wealth$chisq,
  df = length(logrank_wealth$n) - 1
)

logrank_wealth
p_wealth


#--------------------------------------------------
# e) Marital status
#--------------------------------------------------

logrank_marital <- survdiff(
  surv_obj ~ marital_status,
  data = df_surv
)

p_marital <- 1 - pchisq(
  logrank_marital$chisq,
  df = length(logrank_marital$n) - 1
)

logrank_marital
p_marital


#--------------------------------------------------
# f) Religion
#--------------------------------------------------

logrank_religion <- survdiff(
  surv_obj ~ religion,
  data = df_surv
)

p_religion <- 1 - pchisq(
  logrank_religion$chisq,
  df = length(logrank_religion$n) - 1
)

logrank_religion
p_religion


#--------------------------------------------------
# g) Region
#--------------------------------------------------

logrank_region <- survdiff(
  surv_obj ~ region,
  data = df_surv
)

p_region <- 1 - pchisq(
  logrank_region$chisq,
  df = length(logrank_region$n) - 1
)

logrank_region
p_region


#--------------------------------------------------
# h) TV exposure
#--------------------------------------------------

logrank_tv <- survdiff(
  surv_obj ~ tv_exposure,
  data = df_surv
)

p_tv <- 1 - pchisq(
  logrank_tv$chisq,
  df = length(logrank_tv$n) - 1
)

logrank_tv
p_tv


#--------------------------------------------------
# i) Radio exposure
#--------------------------------------------------

logrank_radio <- survdiff(
  surv_obj ~ radio_exposure,
  data = df_surv
)

p_radio <- 1 - pchisq(
  logrank_radio$chisq,
  df = length(logrank_radio$n) - 1
)

logrank_radio
p_radio


#--------------------------------------------------
# j) HIV testing
#--------------------------------------------------

logrank_hiv <- survdiff(
  surv_obj ~ hiv_testing,
  data = df_surv
)

p_hiv <- 1 - pchisq(
  logrank_hiv$chisq,
  df = length(logrank_hiv$n) - 1
)

logrank_hiv
p_hiv

logrank_results <- data.frame(
  Variable = c(
    "Age group",
    "Residence",
    "Education",
    "Wealth",
    "Marital Status",
    "Religion",
    "Region",
    "TV Exposure",
    "Radio Exposure",
    "HIV Testing"
  ),
  P_Value = c(
    p_age,
    p_residence,
    p_education,
    p_wealth,
    p_marital,
    p_religion,
    p_region,
    p_tv,
    p_radio,
    p_hiv
  )
)

logrank_results

###comment###
#Log-rank tests were conducted to assess whether the timing of male circumcision differed across 
#socio-demographic and behavioural characteristics. Significant differences in circumcision timing
#were observed across age group (p = 0.030), residence (p < 0.001), education level (p < 0.001),
#wealth index (p < 0.001), marital status (p < 0.001), religion (p < 0.001), region (p < 0.001), 
#television exposure (p < 0.001), radio exposure (p < 0.001), and HIV testing status (p < 0.001). 
#These findings suggest that the age at which circumcision occurs varies significantly across population subgroups.

########################################################
# Prepare factors for Cox regression
########################################################

df_surv <- df_surv %>%
  mutate(
    tv_exposure = droplevels(tv_exposure),
    radio_exposure = droplevels(radio_exposure)
  )

# Rebuild survey design after updating factor levels
dhs_design_surv <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight_norm,
  data = df_surv,
  nest = TRUE
)

# Verify levels
levels(df_surv$tv_exposure)
levels(df_surv$radio_exposure)

########################################################
# 10# SURVEY-WEIGHTED COX REGRESSION
########################################################

########################################################
# I. UNIVARIABLE COX MODELS
########################################################

# a) Age group
cox_age <- svycoxph(
  Surv(time, event) ~ age_group,
  design = dhs_design_surv
)

summary(cox_age)

broom::tidy(
  cox_age,
  exponentiate = TRUE,
  conf.int = TRUE
)

# b) Residence
cox_residence <- svycoxph(
  Surv(time, event) ~ residence,
  design = dhs_design_surv
)

summary(cox_residence)

broom::tidy(
  cox_residence,
  exponentiate = TRUE,
  conf.int = TRUE
)

# c) Education
cox_education <- svycoxph(
  Surv(time, event) ~ education,
  design = dhs_design_surv
)

summary(cox_education)

broom::tidy(
  cox_education,
  exponentiate = TRUE,
  conf.int = TRUE
)

# d) Wealth
cox_wealth <- svycoxph(
  Surv(time, event) ~ wealth,
  design = dhs_design_surv
)

summary(cox_wealth)

broom::tidy(
  cox_wealth,
  exponentiate = TRUE,
  conf.int = TRUE
)

# e) Marital status
cox_marital <- svycoxph(
  Surv(time, event) ~ marital_status,
  design = dhs_design_surv
)

summary(cox_marital)

broom::tidy(
  cox_marital,
  exponentiate = TRUE,
  conf.int = TRUE
)

# f) Religion
cox_religion <- svycoxph(
  Surv(time, event) ~ religion,
  design = dhs_design_surv
)

summary(cox_religion)

broom::tidy(
  cox_religion,
  exponentiate = TRUE,
  conf.int = TRUE
)

# g) Region
cox_region <- svycoxph(
  Surv(time, event) ~ region,
  design = dhs_design_surv
)

summary(cox_region)

broom::tidy(
  cox_region,
  exponentiate = TRUE,
  conf.int = TRUE
)

# h) TV exposure
cox_tv <- svycoxph(
  Surv(time, event) ~ tv_exposure,
  design = dhs_design_surv
)

summary(cox_tv)

broom::tidy(
  cox_tv,
  exponentiate = TRUE,
  conf.int = TRUE
)

# i) Radio exposure
cox_radio <- svycoxph(
  Surv(time, event) ~ radio_exposure,
  design = dhs_design_surv
)

summary(cox_radio)

broom::tidy(
  cox_radio,
  exponentiate = TRUE,
  conf.int = TRUE
)

# j) HIV testing
cox_hiv <- svycoxph(
  Surv(time, event) ~ hiv_testing,
  design = dhs_design_surv
)

summary(cox_hiv)

broom::tidy(
  cox_hiv,
  exponentiate = TRUE,
  conf.int = TRUE
)


########################################################
# INTERPRETATION OF HAZARD RATIOS
########################################################

# HR > 1  = earlier circumcision
# HR < 1  = later circumcision
# HR = 1  = no difference

# Example:
# HR = 1.30 means a 30% higher hazard of circumcision
# (circumcision tends to occur earlier)

# HR = 0.70 means a 30% lower hazard of circumcision
# (circumcision tends to occur later)

########################################################
# II. MULTIVARIABLE COX MODEL
########################################################

cox_multi <- svycoxph(
  Surv(time, event) ~
    age_group +
    residence +
    education +
    wealth +
    marital_status +
    religion +
    region +
    tv_exposure +
    radio_exposure +
    hiv_testing,
  design = dhs_design_surv
)

summary(cox_multi)

multi_results <- broom::tidy(
  cox_multi,
  exponentiate = TRUE,
  conf.int = TRUE
)

multi_results

write.csv(
  multi_results,
  "multivariable_cox_results.csv",
  row.names = FALSE
)

levels(df_surv$education)

levels(df_surv$wealth)

levels(df_surv$marital_status)

levels(df_surv$religion)

levels(df_surv$region)

###comment###
# Multivariable survey-weighted Cox model summary:
# - Older age groups (25+ years) had significantly lower hazards compared with ages 15–24.
# - Wealth was positively associated with the hazard; richer groups had higher hazards than the poorest.
# - Living with a partner and weekly TV exposure were associated with higher hazards.
# - Animist religion and residence in Far-North and North regions were associated with lower hazards.
# - North-West, West, South-West, Douala, Littoral, and South regions showed higher hazards than Adamawa.
# - Residence, radio exposure, HIV testing, and most education, marital status, and religion categories were not significant after adjustment.
# - Overall model was highly significant (Wald test p < 0.001).

############################################
# 11. PROPORTIONAL HAZARDS ASSUMPTION
############################################

# Fit an unweighted Cox model with the same covariates
cox_ph_check <- coxph(
  Surv(time, event) ~
    age_group +
    residence +
    education +
    wealth +
    marital_status +
    religion +
    region +
    tv_exposure +
    radio_exposure +
    hiv_testing,
  data = df_surv
)

# Schoenfeld residual test
ph_test <- cox.zph(cox_ph_check)

# Display results
ph_test

# Plot Schoenfeld residuals
plot(ph_test)

# Proportional hazards assumption was assessed using Schoenfeld residuals.
# The global test was significant (p < 0.001), indicating violation of the
# proportional hazards assumption. Several covariates, including age group,
# residence, education, wealth, marital status, religion, region, and media
# exposure variables showed evidence of non-proportional hazards, whereas
# HIV testing satisfied the assumption.

# The proportional hazards assumption was assessed using
# Schoenfeld residuals from an equivalent Cox model.
# The global test was significant (χ² = 1178.43, df = 39,
# p < 0.001), indicating evidence of non-proportional hazards.
# Given the large sample size and the objective of identifying
# factors associated with time to first HIV testing, the
# survey-weighted Cox model was retained and results were
# interpreted with caution.

#The proportional hazards assumption was assessed using Schoenfeld residuals. 
#Although the global test indicated statistical evidence of non-proportionality,
#the Cox model was retained because of its robustness and interpretability in large survey datasets.


##################################################
# 12. MODEL DIAGNOSTICS
##################################################

# Martingale residuals
martingale_res <- residuals(
  cox_multi,
  type = "martingale"
)

summary(martingale_res)

# Deviance residuals
deviance_res <- residuals(
  cox_multi,
  type = "deviance"
)

summary(deviance_res)

# Histogram of deviance residuals
hist(
  deviance_res,
  main = "Distribution of Deviance Residuals",
  xlab = "Deviance Residuals"
)

# Potential outliers
which(abs(deviance_res) > 3)

length(which(abs(deviance_res) > 3))

# Model diagnostics were assessed using martingale and deviance residuals.
# Deviance residuals were generally centered around zero, indicating an
# acceptable model fit. Seventy-one observations (1.1% of the sample)
# had absolute deviance residuals greater than 3 and were identified as
# potential outliers. These observations were retained because they likely
# reflect genuine variability in the timing of first HIV testing rather
# than data errors. Overall, the residual diagnostics suggested an
# adequate fit of the survey-weighted Cox model.

################################################
# 13. SENSITIVITY ANALYSIS
# Comparing:
# Analysis A = age_circumcision 95 recoded to 2
# Analysis B = age_circumcision 95 removed
################################################

# Create Analysis B dataset
df_B <- df %>%
  filter(age_circumcision != 95)

# Create survival variables
df_B <- df_B %>%
  mutate(
    time = ifelse(event == 1, age_circumcision, current_age),
    time = ifelse(time == 0, 0.5, time),
    weight_norm = weight / 1000000
  )

df_B <- df_B %>%
  mutate(
    tv_exposure = droplevels(tv_exposure),
    radio_exposure = droplevels(radio_exposure)
  )

# Create survey design
dhs_design_B <- svydesign(
  ids = ~psu,
  strata = ~strata,
  weights = ~weight_norm,
  data = df_B,
  nest = TRUE
)

################################################
# Kaplan-Meier comparison
################################################

km_A <- svykm(
  Surv(time, event) ~ 1,
  design = dhs_design_surv
)

km_B <- svykm(
  Surv(time, event) ~ 1,
  design = dhs_design_B
)

plot(
  km_A,
  col = 1,
  lty = 1,
  xlab = "Age",
  ylab = "Survival probability",
  main = "Sensitivity Analysis: KM Comparison"
)

lines(km_B, col = 2, lty = 2)

legend(
  "bottomleft",
  legend = c(
    "Analysis A (95 recoded to 2)",
    "Analysis B (95 removed)"
  ),
  col = c(1, 2),
  lty = c(1, 2)
)

################################################
# Cox model comparison
################################################

# Analysis A (main model)
cox_A <- cox_multi

# Analysis B
cox_B <- svycoxph(
  Surv(time, event) ~
    age_group +
    residence +
    education +
    wealth +
    marital_status +
    religion +
    region +
    tv_exposure +
    radio_exposure +
    hiv_testing,
  design = dhs_design_B
)

summary(cox_B)

################################################
# Comparison table
################################################

A_results <- broom::tidy(
  cox_A,
  exponentiate = TRUE,
  conf.int = TRUE
)

B_results <- broom::tidy(
  cox_B,
  exponentiate = TRUE,
  conf.int = TRUE
)

A_results$Model <- "Analysis A"
B_results$Model <- "Analysis B"

sensitivity_table <- bind_rows(
  A_results,
  B_results
)

sensitivity_table

write.csv(
  sensitivity_table,
  "sensitivity_analysis_comparison.csv",
  row.names = FALSE
)

#### Comment ####
# Sensitivity analysis comparing Analysis A (95 recoded to age 2)
# and Analysis B (95 removed) showed broadly similar findings.
# The direction and magnitude of key associations, particularly for
# age groups and regions, remained largely unchanged.
# Some variables (education, marital status, TV exposure, and HIV testing)
# lost statistical significance after excluding observations coded as 95,
# indicating a degree of sensitivity for these covariates.
# Overall, the main study conclusions were not materially altered,
# suggesting reasonable robustness of the results to the treatment of
# age_circumcision coded as 95.
