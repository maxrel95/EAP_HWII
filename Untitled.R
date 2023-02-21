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


dataTable = as.data.frame( fread( "test.csv" ) ) # import data
dataTable$er = dataTable$er
fpmg <- pmg(er~  1 +GP + logbm + logme + reversal + mom, dataTable, index=c("jdate","permno"))
summary( fpmg )
stargazer( fpmg, out = 'testss.tex', digits = 3 )



