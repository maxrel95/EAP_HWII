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


dataTable = as.data.frame( fread( "Data/benchmark2.csv" ) ) # import data
dataTable$er = dataTable$er*100
fpmg <- pmg(er~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg )

dataTable$residff3 = dataTable$residff3*100
fpmg2 <- pmg(residff3~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg2 )

dataTable$residff6 = dataTable$residff6*100
fpmg3 <- pmg(residff6~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg3 )

m <- floor(0.75 * nrow(dataTable)^(1/3))
NW_VCOV <- NeweyWest(pmg(er~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno")), 
                     lag = m - 1, prewhite = F, 
                     adjust = T)
coeftest(example_mod, vcov = NW_VCOV)


stargazer( fpmg, out = 'testss.tex', digits = 3 )



