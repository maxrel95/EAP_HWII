# Empirical Asset Pricing 
# Homework II
# Author : Maxime Borel 

library(tidyverse)
library(RSQLite)
library(lubridate)
library(sandwich)
library(broom)
library(dplyr)
library(fixest)
library(data.table) 
library( plm )
library( stargazer )
require(lmtest)


dataTable = as.data.frame( fread( "Data/benchmark23.csv" ) ) # import data
dataTable$er = dataTable$er*100
fpmg <- pmg(er ~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"), )
summary( fpmg )

dataTable$residff3 = dataTable$residff3*100
fpmg2 <- pmg(residff3~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg2 )

dataTable$residff6 = dataTable$residff6*100
fpmg3 <- pmg(residff6~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg3 )

stargazer( fpmg, fpmg2, fpmg3, out = 'testss.tex', digits = 3 )


dataTable2 = as.data.frame( fread( "Data/benchmark2_q.csv" ) ) # import data
dataTable2$er = dataTable2$er*100
fpmg_q <- pmg(er~  1 +GP + logbm + logme + reversal + mom, dataTable2, index=c("jdate","permno"))
summary( fpmg_q )
