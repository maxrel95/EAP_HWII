# Empirical Asset Pricing 
# Homework II
# Author : Maxime Borel 

library(dplyr)
library(data.table) 
library( plm )
library( stargazer )
library(lmtest)
library(sandwich)


dataTable = as.data.frame( fread( "Data/benchmark.csv" ) ) # import data
dataTable$er = dataTable$er*100
fpmg <- pmg(er ~  1 +GP + logbm + logme1 + reversal + mom, dataTable, index=c("jdate","permno"), model = "dmg" )
fpmg1 <- pmg(er ~  1  + logbm + logme1 + reversal + mom, dataTable, index=c("jdate","permno") )

summary( fpmg )
summary( fpmg1 )

stargazer( fpmg1,fpmg, out = 'results/fmregression_replication.tex', digits = 2,
           title="Regression Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER"), styles = 'aer', report = 'vc*t',
           covariate.labels=c("Gross Profit", "log(BE/ME)","log(ME)","r(1,1)","r(12,2)"))

# Full annually with excess return 
df_a_full = as.data.frame( fread( "Data/all_df.csv" ) )
df_a_full$er = df_a_full$er*100
fm_a_full_er <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_full, index=c("jdate","permno") )
summary( fm_a_full_er )

# same basis as quarterly 
df_a_full_filtered = df_a_full %>%
  filter( jdate >='1975-07-30' )
fm_a_full_filtered <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_full_filtered, index=c("jdate","permno") )
summary( fm_a_full_filtered )

# Full annually with ff3
df_a_full$residff3 = df_a_full$residff3*100
fm_a_full_ff3 <- pmg(residff3 ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_full, index=c("jdate","permno") )
summary( fm_a_full_ff3 )

# Full annually with ff3
df_a_full$residff6 = df_a_full$residff6*100
fm_a_full_ff6 <- pmg(residff6 ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_full, index=c("jdate","permno") )
summary( fm_a_full_ff6 )

# Large annually with er
df_a_large = df_a_full %>% 
  filter( szport == 'Large' )
fm_a_large_er <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_large, index=c("jdate","permno") )
summary( fm_a_large_er )

# Small annually with er
df_a_small = df_a_full %>% 
  filter( szport == 'Small' )
fm_a_small_er <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_small, index=c("jdate","permno") )
summary( fm_a_small_er )

# Micro annually with er
df_a_micro = df_a_full %>% 
  filter( szport == 'Micro' )
fm_a_micro_er <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_a_micro, index=c("jdate","permno") )
summary( fm_a_micro_er )

# Exclude Fin/Util annually with er
df_a_indu = df_a_full %>% 
  filter( ( siccd < 6000 | siccd > 6799 ) & ( siccd < 9000 | siccd > 9799 ) )
fm_a_indu_er <- pmg(er ~1 + GP + gat + logbm + logme1 + reversal + mom, df_a_indu, index=c("jdate","permno") )
summary( fm_a_indu_er )

# Full quarterly
df_q_full = as.data.frame( fread( "Data/all_df_q.csv" ) )

# get numbers of obs per year
nbrOfObsPerYear = df_q_full %>% 
  group_by( jdate ) %>%
  summarise( n = n() )

df_q_full_filtered = df_q_full %>%
  filter( jdate >='1975-07-30' )
df_q_full_filtered$er = df_q_full_filtered$er*100
fm_q_full <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_q_full_filtered, index=c("jdate","permno") )
summary( fm_q_full )

# not reported but they dont give more value to display them 
# Full quarterly  with ff3 
df_q_full_filtered$residff6 = df_q_full_filtered$residff6*100
fm_q_full_ff6 <- pmg(residff6 ~ 1 + GP + gat + logbm + logme1 + reversal + mom, df_q_full_filtered, index=c("jdate","permno") )
summary( fm_q_full_ff6 )

# quarterly without financial and utilities 
df_q_indu = df_q_full_filtered %>% 
  filter( ( siccd < 6000 | siccd > 6799 ) & ( siccd < 9000 | siccd > 9799 ) )
fm_q_indu_er <- pmg(er ~1 + GP + gat + logbm + logme1 + reversal + mom, df_q_indu, index=c("jdate","permno") )
summary( fm_q_indu_er )
#trace(stargazer:::.stargazer.wrap, edit = T)

stargazer( fm_a_full_er, fm_a_full_ff6, fm_q_full, fm_a_indu_er, out = 'results/fmregression.tex', digits = 2,
           title="Regression Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER", "FF6", "ER-Q", "ER-Ind"), styles = 'aer', report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","r(1,1)","r(12,2)"))

stargazer( fm_a_full_er, fm_a_large_er, fm_a_small_er, fm_a_micro_er, out = 'results/fmregressionSize.tex',
           digits = 2, styles = 'aer', report = 'vc*t',
           title="Regression Results", align=TRUE, dep.var.labels=c("All","Large", "Small", "Micro"),
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","r(1,1)","r(12,2)"))

stargazer( fm_a_full_er, fm_a_full_ff6, fm_q_full, fm_a_indu_er, type = 'text', digits = 2,
           title="Regression Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER", "FF6", "ER-Q", "ER-Ind"), styles = 'aer', report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","r(1,1)","r(12,2)"))

stargazer( fm_a_full_er, fm_a_large_er, fm_a_small_er, fm_a_micro_er, type = 'text',
           digits = 2,
           title="Regression Results", align=TRUE, dep.var.labels=c("All","Large", "Small", "Micro"), report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","r(1,1)","r(12,2)"))

# interactions 
interaction_df = df_a_full %>%
  filter( szport !='Micro' )

gp_gat <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + GP*gat, interaction_df, index=c("jdate","permno") )
gp_logbm <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + GP*logbm, interaction_df, index=c("jdate","permno") )
gp_logme1 <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + GP*logme1, interaction_df, index=c("jdate","permno") )
gp_reversal <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + GP*reversal, interaction_df, index=c("jdate","permno") )
gp_mom <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + GP*mom, interaction_df, index=c("jdate","permno") )

gat_logbm <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + gat*logbm, interaction_df, index=c("jdate","permno") )
gat_logme1 <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + gat*logme1, interaction_df, index=c("jdate","permno") )
gat_reversal <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + gat*reversal, interaction_df, index=c("jdate","permno") )
gat_mom <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + gat*mom, interaction_df, index=c("jdate","permno") )

logbm_logme1 <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logbm*logme1, interaction_df, index=c("jdate","permno") )
logbm_reversal <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logbm*reversal, interaction_df, index=c("jdate","permno") )
logbm_mom <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logbm*mom, interaction_df, index=c("jdate","permno") )

logme1_reversal <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logme1*reversal, interaction_df, index=c("jdate","permno") )
logme1_mom <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logme1*mom, interaction_df, index=c("jdate","permno") )

reversal_mom <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom, interaction_df, index=c("jdate","permno") )

tstat_int = c(tail(gp_gat$coefficients,1)/sqrt(tail(diag(gp_gat$vcov),1)), tail(gp_logbm$coefficients,1)/sqrt(tail(diag(gp_logbm$vcov),1)),
  tail(gp_logme1$coefficients,1)/sqrt(tail(diag(gp_logme1$vcov),1)), tail(gp_reversal$coefficients,1)/sqrt(tail(diag(gp_reversal$vcov),1)),
  tail(gp_mom$coefficients,1)/sqrt(tail(diag(gp_mom$vcov),1)), tail(gat_logbm$coefficients,1)/sqrt(tail(diag(gat_logbm$vcov),1)),
  tail(gat_logme1$coefficients,1)/sqrt(tail(diag(gat_logme1$vcov),1)), tail(gat_reversal$coefficients,1)/sqrt(tail(diag(gat_reversal$vcov),1)), 
  tail(gat_mom$coefficients,1)/sqrt(tail(diag(gat_mom$vcov),1)), tail(logbm_logme1$coefficients,1)/sqrt(tail(diag(logbm_logme1$vcov),1)),
  tail(logbm_reversal$coefficients,1)/sqrt(tail(diag(logbm_reversal$vcov),1)), tail(logbm_mom$coefficients,1)/sqrt(tail(diag(logbm_mom$vcov),1)), 
  tail(logme1_reversal$coefficients,1)/sqrt(tail(diag(logme1_reversal$vcov),1)), tail(logme1_mom$coefficients,1)/sqrt(tail(diag(logme1_mom$vcov),1)),
  tail(reversal_mom$coefficients,1)/sqrt(tail(diag(reversal_mom$vcov),1)) )
tstat_int[order(abs(tstat_int), decreasing=TRUE)[1:5]]

mom_interaction <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + mom*logbm,
                       df_a_full, index=c("jdate","permno") )
rev_interaction <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + reversal*gat,
                       df_a_full, index=c("jdate","permno") )
bm_interaction <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logbm*mom + logbm*logme1,
                      df_a_full, index=c("jdate","permno") )
all_interaction = pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + mom*logbm + logbm*logme1,
                      df_a_full, index=c("jdate","permno") )

mom_interaction_nomic <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + mom*logbm,
                             interaction_df, index=c("jdate","permno") )
rev_interaction_nomic <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + reversal*gat,
                             interaction_df, index=c("jdate","permno") )
bm_interaction_nomic <- pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + logbm*mom + logbm*logme1,
                            interaction_df, index=c("jdate","permno") )
all_interaction_nomic = pmg(er ~ 1 + GP + gat + logbm + logme1 + reversal + mom + reversal*mom + mom*logbm + logbm*logme1,
                            interaction_df, index=c("jdate","permno") )

stargazer( mom_interaction, bm_interaction, all_interaction,  type = 'text', digits = 2,
           title="Regression with Interaction Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER-All", "ER-noMic"), styles = 'aer', report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","$r_{1,1}$","$r_{12,2}$",
                              "momxrev","momxbm","bmxsize"))

stargazer( mom_interaction, bm_interaction, all_interaction,  out = 'results/interactions.tex', digits = 2,
           title="Regression with Interaction Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER"), styles = 'aer', report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","$r_{1,1}$","$r_{12,2}$",
                               "$r_{12,2}$ X $r_{1,1}$","$r_{12,2}$ X log(BE/ME)","log(BE/ME) X log(ME)"))

stargazer( mom_interaction_nomic, bm_interaction_nomic, all_interaction_nomic,  out = 'results/interactions_mic.tex', digits = 2,
           title="Regression with Interaction Results", digits.extra = 1,
           align=TRUE, dep.var.labels=c("ER"), report = 'vc*t',
           covariate.labels=c("Gross Profit","Asset Growth", "log(BE/ME)","log(ME)","$r_{1,1}$","$r_{12,2}$",
                              "$r_{12,2}$ X $r_{1,1}$","$r_{12,2}$ X log(BE/ME)","log(BE/ME) X log(ME)"))

