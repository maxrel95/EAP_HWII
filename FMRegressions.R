# Empirical Asset Pricing 
# Homework II
# Author : Maxime Borel 

library(dplyr)
library(data.table) 
library( plm )
library( stargazer )

# Full annually with excess return 
df_a_full = as.data.frame( fread( "Data/all_df.csv" ) )
df_a_full$er = df_a_full$er*100
fm_a_full_er <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno") )
summary( fm_a_full_er )

# Full annually with ff6
df_a_full$residff6 = df_a_full$residff6*100
fm_a_full_ff6 <- pmg(residff6 ~  1 + GP + gat + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno") )
summary( fm_a_full_ff6 )

# Large annually with er
df_a_large = df_a_full %>% 
  filter( szport = 'Large' )
fm_a_large_er <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, df_a_large, index=c("jdate","permno") )
summary( fm_a_large_er )

# Small annually with er
df_a_small = df_a_full %>% 
  filter( szport = 'Small' )
fm_a_small_er <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, df_a_small, index=c("jdate","permno") )
summary( fm_a_small_er )

# Micro annually with er
df_a_micro = df_a_full %>% 
  filter( szport = 'Micro' )
fm_a_micro_er <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, df_a_micro, index=c("jdate","permno") )
summary( fm_a_micro_er )

# Exclude Fin/Util annually with er
df_a_indu = df_a_full %>% 
  filter( ( siccd < 4900 | siccd > 4999 ) & ( siccd < 6000 | siccd > 6999 ) )
fm_a_indu_er <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, df_a_indu, index=c("jdate","permno") )
summary( fm_a_indu_er )

# Full quarterly
df_q_full = as.data.frame( fread( "Data/all_df_q.csv" ) )
df_q_full$er = df_q_full$er*100
fm_q_full <- pmg(er ~  1 + GP + gat + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"), )
summary( fm_q_full )

