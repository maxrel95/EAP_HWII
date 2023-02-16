# Author : Maxime Borel 
# Class : Empirical Asset Pricing 
# Homework II

import numpy as np 
import pandas as pd
import wrds
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats
import statsmodels.api as sm

db = wrds.Connection( wrds_username = 'maxrel95' )

# Annual fundamental data request
compustat_annual = db.raw_sql("""
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk, gp
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/2015'
                    """, date_cols=['datadate'])
# Link table
link_table = db.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """, date_cols=['linkdt', 'linkenddt'])

# if linkenddt is missing then set to today date
link_table[ 'linkenddt' ] = link_table[ 'linkenddt' ].fillna( pd.to_datetime( 'today' ) )

compustat_annual[ 'jdate' ] = compustat_annual[ 'datadate' ] + MonthEnd( 6 ) # creat a col with year only 

# create preferrerd stock
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'pstkrv' ].isnull(),
 compustat_annual[ 'pstkl' ], compustat_annual[ 'pstkrv' ] )
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'ps' ].isnull(),compustat_annual[ 'pstk' ],
 compustat_annual[ 'ps' ] )
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'ps' ].isnull(), 0, compustat_annual[ 'ps' ])
compustat_annual[ 'txditc' ] = compustat_annual[ 'txditc' ].fillna(0)

# create book equity
compustat_annual[ 'be' ] = compustat_annual[ 'seq' ]+compustat_annual[ 'txditc' ]-compustat_annual[ 'ps' ]
compustat_annual[ 'be' ] = np.where( compustat_annual[ 'be' ]>0, compustat_annual[ 'be' ], np.nan )

compustat_annual[ 'gat' ] = compustat_annual.groupby( [ 'gvkey' ] )[ 'at' ].pct_change()
compustat_annual[ 'GP' ] = compustat_annual[ 'gp'] / compustat_annual.groupby( [ 'gvkey' ] )[ 'at' ].shift( 1 )

# number of years in compustat_annualustat
compustat_annual = compustat_annual.sort_values( by=[ 'gvkey','datadate' ] )

ccm1 = pd.merge( compustat_annual[ [ 'gvkey', 'datadate', 'jdate', 'be', 'at', 'gp', 'gat', 'GP' ] ],
         link_table, how='left', on=['gvkey'] )

# set link date bounds
ccm2 = ccm1[ ( ccm1[ 'jdate' ] >= ccm1[ 'linkdt' ] ) & ( ccm1[ 'jdate' ] <= ccm1[ 'linkenddt' ] ) ]
ccm2 = ccm2[['gvkey','permno','datadate', 'jdate', 'be', 'at', 'gp', 'gat', 'GP' ]] 
ccm2[ 'permno' ] = ccm2[ 'permno' ].astype( int )

crsp_m = db.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc, b.siccd
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/2015' and '12/31/2021'
                      and b.exchcd between 1 and 3
                      """, date_cols=['date']) 

# change variable format to int
crsp_m[ [ 'permco','permno','shrcd','exchcd', 'siccd' ] ] = crsp_m[ [ 'permco','permno','shrcd','exchcd',
     'siccd' ] ].astype( int )

# Line up date to be end of month
crsp_m[ 'jdate' ] = crsp_m[ 'date' ] + MonthEnd( 0 )

# add delisting return
#table monthly stock event delisting
dlret = db.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """, date_cols=['dlstdt'])

dlret.permno = dlret.permno.astype( int )
dlret[ 'jdate' ] = dlret[ 'dlstdt' ] + MonthEnd( 0 )

crsp = pd.merge( crsp_m, dlret, how='left',on=['permno','jdate'] )
crsp['dlret'] = crsp['dlret'].fillna(0)
crsp['ret'] = crsp['ret'].fillna(0)

# retadj factors in the delisting returns
crsp['retadj'] = ( 1+crsp['ret'])*(1+crsp['dlret'] )-1

# calculate market equity
crsp[ 'me' ] = crsp[ 'prc' ].abs()*crsp[ 'shrout' ] 
crsp = crsp.drop( [ 'dlret', 'dlstdt', 'prc', 'shrout' ], axis=1)
crsp = crsp.sort_values(by=['jdate', 'permco', 'me'])

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate','permco'])[ 'me' ].sum().reset_index()

# largest mktcap within a permco/date
crsp_maxme = crsp.groupby( [ 'jdate', 'permco' ] )[ 'me' ].max().reset_index()

# join by jdate/maxme to find the permno
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['jdate','permco','me'])

# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)

# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=[ 'jdate', 'permco' ] )

# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values( by=['permno', 'jdate'] ).drop_duplicates()

# lag 6 month the market cap
me = crsp2[ [ 'permno', 'jdate', 'me'] ] 
me[ 'jdate' ] = me[ 'jdate' ] + MonthEnd( 6 )
me.rename( columns={ 'me': 'lag6_me'}, inplace=True )
crsp3 = pd.merge( crsp2, me, how='left', on=[ 'permno', 'jdate' ] )

reversal  = crsp2[ [ 'permno', 'jdate', 'retadj'] ]
reversal[ 'jdate' ] = reversal[ 'jdate' ] + MonthEnd( 1 )
reversal.rename( columns={ 'retadj': 'reversal'}, inplace=True )
crsp4 = pd.merge( crsp3, reversal, how='left', on=['permno', 'jdate'] )

mom = crsp2[ [ 'permno', 'jdate', 'ret'] ]
mom[ 'gross_ret' ] = 1 + mom[ 'ret' ]
mom[ 'mom' ] = mom.groupby( ['permno'] )[ 'gross_ret' ].rolling( window=11, closed='left' ).apply( 
    lambda x: x.prod() ).reset_index( 0, drop=True ) - 1
mom = mom[ ['permno', 'jdate', 'ret', 'mom'] ]
crsp5 = pd.merge( crsp4, mom, how='left', on=['permno', 'jdate'] )

## merged link and fundamental 
df = pd.merge_ordered( crsp5, ccm2, how='left', on=[ 'permno', 'jdate' ], fill_method='ffill' )
df[ 'beme' ] = df[ 'be' ]*1000 / df[ 'lag6_me' ]

annual_df = df[ ['permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retx', 'me',
                 'lag6_me', 'reversal', 'mom', 'gat', 'GP', 'beme'] ]
annual_df[ 'count' ] = annual_df.groupby( ['permno'] ).cumcount()

nyse = annual_df[ ( annual_df['beme']>0 ) & ( annual_df['me']>0 ) & \
             ( annual_df['count']>=1 ) & ( ( annual_df['shrcd']==10 ) | ( annual_df['shrcd']==11 ) )]

nyse_sz = nyse.groupby( ['jdate'] )[ 'me' ].describe( percentiles=[ 0.25, 0.75 ] ).reset_index()
nyse_sz = nyse_sz[ ['jdate','25%','75%'] ].rename( columns={ '25%':'sz25', '75%':'sz75' } )

def sz_bucket( row ):
    if row[ 'me' ]<=row[ 'sz25' ]:
        value = 'S'
    elif row[ 'me' ]<=row[ 'sz75' ]:
        value ='R'
    elif row[ 'me' ]>row[ 'sz75' ]:
        value = 'B'
    else:
        value = ''    
    return value
