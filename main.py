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
compustat_annual[ 'at' ] = np.where( compustat_annual['at'] == 0, np.nan, compustat_annual['at'] )


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

# download price 
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
crsp = crsp.drop( [ 'dlret', 'dlstdt', 'prc', 'shrout' ], axis=1 )
crsp = crsp.sort_values( by=['jdate', 'permco', 'me'] )

### Aggregate Market Cap ###
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby( ['jdate','permco'] )[ 'me' ].sum().reset_index()

# largest mktcap within a permco/date
crsp_maxme = crsp.groupby( [ 'jdate', 'permco' ] )[ 'me' ].max().reset_index()

# join by jdate/maxme to find the permno
crsp1 = pd.merge( crsp, crsp_maxme, how='inner', on=['jdate','permco','me'] )

# drop me column and replace with the sum me
crsp1 = crsp1.drop( ['me'], axis=1 )

# join with sum of me to get the correct market cap info
crsp2 = pd.merge( crsp1, crsp_summe, how='inner', on=[ 'jdate', 'permco' ] )

# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values( by=['permno', 'jdate'] ).drop_duplicates()

# lag 6 month the market cap
me = crsp2[ [ 'permno', 'jdate', 'me'] ] 
me[ 'jdate' ] = me[ 'jdate' ] + MonthEnd( 6 )
me.rename( columns={ 'me': 'lag6_me'}, inplace=True )
crsp3 = pd.merge( crsp2, me, how='left', on=[ 'permno', 'jdate' ] )

reversal  = crsp2[ [ 'permno', 'jdate', 'ret'] ]
reversal[ 'jdate' ] = reversal[ 'jdate' ] + MonthEnd( 1 )
reversal.rename( columns={ 'ret': 'reversal'}, inplace=True )
crsp4 = pd.merge( crsp3, reversal, how='left', on=['permno', 'jdate'] )

mom = crsp2[ [ 'permno', 'jdate', 'ret'] ]
mom[ 'gross_ret' ] = 1 + mom[ 'ret' ]
mom[ 'mom' ] = mom.groupby( ['permno'] )[ 'gross_ret' ].rolling( window=11, closed='left' ).apply( 
    lambda x: x.prod() ).reset_index( 0, drop=True ) - 1
mom = mom[ ['permno', 'jdate', 'ret', 'mom'] ]
crsp5 = pd.merge( crsp4, mom, how='left', on=['permno', 'jdate'] )

## merged link and fundamental 
df[ 'beme' ] = df[ 'be' ]*1000 / df[ 'lag6_me' ]

annual_df = df[ ['permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retx', 'me',
                 'lag6_me', 'reversal', 'mom', 'gat', 'GP', 'beme'] ]
annual_df[ 'count' ] = annual_df.groupby( ['permno'] ).cumcount()

nyse = annual_df[ ( annual_df['beme']>0 ) & ( annual_df['me']>0 ) & \
             ( annual_df['count']>=1 ) & ( ( annual_df['shrcd']==10 ) | ( annual_df['shrcd']==11 ) )]

nyse_sz = nyse.groupby( ['jdate'] )[ 'me' ].describe( percentiles=[ 0.3, 0.7 ] ).reset_index()
nyse_sz = nyse_sz[ ['jdate','30%','70%'] ].rename( columns={ '30%':'sz30', '70%':'sz70' } )

annual_df = pd.merge( annual_df, nyse_sz, how='inner', on=[ 'jdate'] )

def sz_bucket( row ):
    if row[ 'me' ]<=row[ 'sz30' ]:
        value = 'Micro'
    elif row[ 'me' ]<=row[ 'sz70' ]:
        value ='Small'
    elif row[ 'me' ]>row[ 'sz70' ]:
        value = 'Large'
    else:
        value = ''    
    return value


annual_df['szport'] = np.where( (annual_df['beme']>0) & (annual_df['me']>0) & (annual_df['count']>=1),
                                 annual_df.apply(sz_bucket, axis=1), '')
annual_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

# winsorized the data 
bm_trim = nyse.groupby( 'jdate' )[ 'beme' ].describe( percentiles=[ .01, .99 ] ).reset_index()
bm_trim = bm_trim[ ['jdate','1%','99%'] ]

annual_df[ 'beme2' ] = np.where( annual_df[ 'beme' ]>=bm_trim['99%'], bm_trim['99%'],  annual_df[ 'beme' ])
com = annual_df[['permno', 'jdate', 'beme', 'beme2']]
annual_df_large = annual_df[ annual_df[ 'szport' ] == 'Large' ]
annual_df_small = annual_df[ annual_df[ 'szport' ] == 'Small' ]
annual_df_micro = annual_df[ annual_df[ 'szport' ] == 'Micro' ]

annual_df_noFinUt = annual_df[ ( ( annual_df['siccd'] > 4900 ) & ( annual_df['siccd'] <= 4949 ) ) |
                               ( ( annual_df['siccd'] > 6000 ) & ( annual_df['siccd'] <= 6799 ) ) ]


# quarterly data
def aggregate_quarter(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm4
    """
    lag = pd.DataFrame()
    for i in range(1, 4):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby( 'gvkey' )['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series]
    return result

compustat_q = db.raw_sql("""
                    select gvkey, datadate, atq, pstkq, txditcq,
                    seqq, revtq, cogsq
                    from comp.fundq
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/2015'
                    """, date_cols=['datadate'])

compustat_q[ 'jdate' ] = compustat_q[ 'datadate' ] + MonthEnd( 4 ) # creat a col with year only 

# number of years in compustat_annualustat
compustat_q = compustat_q.sort_values( by=[ 'gvkey','datadate' ] )


compustat_q[ 'beq' ] = np.where( compustat_q[ 'seqq' ]>0, compustat_q['seqq'] +
                                 compustat_q['txditcq'] - compustat_q['pstkq'], np.nan )
compustat_q[ 'beq' ] = np.where( compustat_q[ 'beq' ]<=0, np.nan, compustat_q['beq'] )
compustat_q[ 'beq_lag4' ] = compustat_q.groupby( ['gvkey'] )[ 'beq' ].shift( 4 )  


compustat_q[ 'atq' ] = np.where( compustat_q[ 'atq' ] == 0, np.nan, compustat_q['atq'] )
compustat_q[ 'atq_lag4' ] = compustat_q.groupby( [ 'gvkey' ] )[ 'atq' ].shift( 4 )
compustat_q[ 'atq_lag1' ] = compustat_q.groupby( [ 'gvkey' ] )[ 'atq' ].shift( 1 )
compustat_q[ 'gat' ] = ( compustat_q[ 'atq' ] - compustat_q[ 'atq_lag4' ] ) / compustat_q[ 'atq_lag4' ]


compustat_q['revtq4'] = aggregate_quarter( 'revtq', compustat_q )
compustat_q['cogsq4'] = aggregate_quarter( 'cogsq', compustat_q )
compustat_q['GP'] = ( compustat_q[ 'revtq4' ] - compustat_q[ 'cogsq4' ] ) / compustat_q[ 'atq_lag4' ]


ccm1_q = pd.merge( compustat_q[ [ 'gvkey', 'datadate', 'jdate', 'beq_lag4', 'gat', 'GP' ] ],
                   link_table, how='left', on=['gvkey'] )

# set link date bounds
ccm2_q = ccm1_q[ ( ccm1_q[ 'jdate' ] >= ccm1_q[ 'linkdt' ] ) & ( ccm1_q[ 'jdate' ] <= ccm1_q[ 'linkenddt' ] ) ]
ccm2_q = ccm2_q[['gvkey','permno','datadate', 'jdate', 'beq_lag4', 'gat', 'GP' ]] 
ccm2_q[ 'permno' ] = ccm2_q[ 'permno' ].astype( int )

# lag 6 month the market cap
me_q = crsp2[ [ 'permno', 'jdate', 'me'] ] 
me_q[ 'jdate' ] = me_q[ 'jdate' ] + MonthEnd( 4 )
me_q.rename( columns={ 'me': 'lag4_me'}, inplace=True )
crsp3_q = pd.merge( crsp2, me_q, how='left', on=[ 'permno', 'jdate' ] )

crsp4_q = pd.merge( crsp3_q, reversal, how='left', on=['permno', 'jdate'] )

crsp5_q = pd.merge( crsp4_q, mom, how='left', on=['permno', 'jdate'] )

## merged link and fundamental 
df_q = pd.merge_ordered( crsp5_q, ccm2_q, how='left', on=[ 'permno', 'jdate' ], fill_method='ffill' )
df_q[ 'beme' ] = df_q[ 'beq_lag4' ]*1000 / df_q[ 'lag4_me' ]

quarter_df = df_q[ ['permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retx', 'me',
                   'lag4_me', 'reversal', 'mom', 'gat', 'GP', 'beme' ] ]
quarter_df[ 'count' ] = quarter_df.groupby( ['permno'] ).cumcount()

nyse_q = quarter_df[ ( quarter_df['beme']>0 ) & ( quarter_df['me']>0 ) & \
                  ( quarter_df['count']>=1 ) & ( ( quarter_df['shrcd']==10 ) | ( quarter_df['shrcd']==11 ) )]

nyse_sz_q = nyse_q.groupby( ['jdate'] )[ 'me' ].describe( percentiles=[ 0.3, 0.7 ] ).reset_index()
nyse_sz_q = nyse_sz_q[ ['jdate','30%','70%'] ].rename( columns={ '30%':'sz30', '70%':'sz70' } )

quarter_df = pd.merge( quarter_df, nyse_sz_q, how='inner', on=[ 'jdate'] )

def sz_bucket( row ):
    if row[ 'me' ]<=row[ 'sz30' ]:
        value = 'Micro'
    elif row[ 'me' ]<=row[ 'sz70' ]:
        value ='Small'
    elif row[ 'me' ]>row[ 'sz70' ]:
        value = 'Large'
    else:
        value = ''    
    return value


quarter_df['szport'] = np.where( (quarter_df['beme']>0) & (quarter_df['me']>0) & (quarter_df['count']>=1),
                                 quarter_df.apply( sz_bucket, axis=1 ), '')
quarter_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

quarter_df_large = quarter_df[ quarter_df[ 'szport' ] == 'Large' ]
quarter_df_small = quarter_df[ quarter_df[ 'szport' ] == 'Small' ]
quarter_df_micro = quarter_df[ quarter_df[ 'szport' ] == 'Micro' ]

## Fama-McBeth regression 












