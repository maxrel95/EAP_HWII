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
from linearmodels import FamaMacBeth


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
                    and datadate >= '01/01/1960'
                    """, date_cols=['datadate'])

compustat_annual = compustat_annual.sort_values( by=['gvkey', 'datadate'] ).drop_duplicates()

compustat_annual[ 'at' ] = np.where( compustat_annual[ 'at' ] == 0, np.nan, compustat_annual[ 'at' ] )
compustat_annual = compustat_annual.dropna( subset=['at'] )

## CRSP block 
crsp = db.raw_sql("""
                      select a.permno, a.permco, a.date, b.shrcd, b.exchcd,
                      a.ret, a.retx, a.shrout, a.prc, b.siccd
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date >= '01/01/1960'
                      and b.exchcd between 1 and 3
                      """, date_cols=['date']) 

crsp[ [ 'permco', 'permno', 'shrcd', 'exchcd', 'siccd' ] ] = crsp[ [ 'permco', 'permno', 'shrcd', 'exchcd',
     'siccd' ] ].astype( int )

# Line up date to be end of month
crsp[ 'jdate' ] = crsp[ 'date' ] + MonthEnd( 0 )

crsp = crsp.dropna( subset=['prc'] )
crsp[ 'me' ] = crsp[ 'prc' ].abs()*crsp[ 'shrout' ] 

# if market cap is nan then let the return equal to 0
crsp[ 'ret' ] = np.where( crsp[ 'me' ].isnull(), 0, crsp[ 'ret' ] )
crsp[ 'retx' ] = np.where( crsp[ 'me' ].isnull(), 0, crsp[ 'retx' ] )

crsp = crsp.sort_values( by=['permno', 'date'] ).drop_duplicates()
crsp[ 'me' ] = np.where( crsp[ 'permno' ] == crsp[ 'permno' ].shift( 1 ), crsp[ 'me' ].fillna( method='ffill' ),
                         crsp[ 'me' ] )

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

## ccm block
link_table = db.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """, date_cols=['linkdt', 'linkenddt'])

# if linkenddt is missing then set to today date
link_table[ 'linkenddt' ] = link_table[ 'linkenddt' ].fillna( pd.to_datetime( 'today' ) )

ccm1 = pd.merge( compustat_annual, link_table, how='left', on=['gvkey'] )

ccm1[ 'yearend' ] = ccm1[ 'datadate' ] + YearEnd( 0 )
ccm1[ 'jdate' ] = ccm1[ 'datadate' ] + MonthEnd( 6 )

ccm2 = ccm1[ ( ccm1[ 'jdate' ] >= ccm1[ 'linkdt' ] ) & ( ccm1[ 'jdate' ] <= ccm1[ 'linkenddt' ] ) ]

df = pd.merge( crsp2, ccm2, how='inner', on=[ 'permno', 'jdate' ] )
df = df[((df['exchcd'] == 1) | (df['exchcd'] == 2) | (df['exchcd'] == 3)) &
                   ((df['shrcd'] == 10) | (df['shrcd'] == 11))]

df[ 'me' ] = np.where( df[ 'me' ] == 0, np.nan, df[ 'me' ] )
df = df.dropna( subset=['me'] )

df.loc[ df.groupby( ['datadate', 'permno', 'linkprim'], as_index=False ).nth( [0] ).index, 'temp' ] = 1
df = df[df['temp'].notna()]
df.loc[ df.groupby( ['permno', 'yearend', 'datadate'], as_index=False ).nth( [-1] ).index, 'temp'] = 1
df = df[ df['temp'].notna() ]

df = df.sort_values( by=['permno', 'jdate'] )

## annual variables
df[ 'ps' ] = np.where( df[ 'pstkrv' ].isnull(),
                                     df[ 'pstkl' ], df[ 'pstkrv' ] )
df[ 'ps' ] = np.where( df[ 'ps' ].isnull(),
                                     df[ 'pstk' ], df[ 'ps' ] )
df[ 'ps' ] = np.where( df[ 'ps' ].isnull(), 0, df[ 'ps' ])

df[ 'txditc' ] = df[ 'txditc' ].fillna(0)

# book equity
df[ 'be' ] = df[ 'seq' ] + df[ 'txditc' ] - df[ 'ps' ]
df[ 'be' ] = np.where( df[ 'be' ]>0, df[ 'be' ], np.nan )

# total asset growth
df[ 'gat' ] = df.groupby( [ 'gvkey' ] )[ 'at' ].pct_change()

# gross profitability
df[ 'GP' ] = df[ 'gp'] / df.groupby( [ 'gvkey' ] )[ 'at' ] #.shift( 1 )


# Momentum 
crsp_mom = db.raw_sql(
                    """
                      select permno, date, ret, retx, shrout, prc
                      from crsp.msf 
                      where date >= '01/01/1960'
                      """, date_cols=['date']
) 

crsp_mom[ 'permno' ] = crsp_mom[ 'permno' ].astype( int )
crsp_mom[ 'jdate' ] = crsp_mom[ 'date' ] + MonthEnd( 0 )
crsp_mom = crsp_mom.dropna( subset=[ 'ret', 'retx', 'prc' ] )

dlret = db.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """, date_cols=['dlstdt'])

dlret.permno = dlret.permno.astype( int )
dlret[ 'jdate' ] = dlret[ 'dlstdt' ] + MonthEnd( 0 )

crsp_mom = pd.merge( crsp_mom, dlret, how='left',on=['permno','jdate'] )
crsp_mom['dlret'] = crsp_mom['dlret'].fillna(0)
crsp_mom['ret'] = crsp_mom['ret'].fillna(0)

# retadj factors in the delisting returns
crsp_mom['retadj'] = ( 1+crsp_mom['ret'])*(1+crsp_mom['dlret'] ) - 1
crsp_mom[ 'me' ] = crsp_mom[ 'prc' ].abs()*crsp_mom[ 'shrout' ] 

mom = crsp_mom[ [ 'permno', 'jdate', 'ret'] ]
mom[ 'gross_ret' ] = 1 + mom[ 'ret' ]
mom[ 'mom' ] = mom.groupby( ['permno'] )[ 'gross_ret' ].rolling( window=11, min_periods=11, closed='left' ).apply( 
    lambda x: x.prod() ).reset_index( 0, drop=True ) - 1
mom = mom[ ['permno', 'jdate', 'mom'] ].sort_values( by=['permno', 'jdate'] )
crsp_mom1 = pd.merge( crsp_mom, mom, how='left', on=['permno', 'jdate'] )

me = crsp_mom[ [ 'permno', 'jdate', 'me'] ] 
me[ 'jdate' ] = me[ 'jdate' ] + MonthEnd( 6 )
me.rename( columns={ 'me': 'lag6_me'}, inplace=True )
crsp_mom2 = pd.merge( crsp_mom1, me, how='left', on=[ 'permno', 'jdate' ] )

me1 = crsp_mom[ [ 'permno', 'jdate', 'me'] ] 
me1[ 'jdate' ] = me1[ 'jdate' ] + MonthEnd( 1 )
me1.rename( columns={ 'me': 'lag1_me'}, inplace=True )
crsp_mom2 = pd.merge( crsp_mom2, me1, how='left', on=[ 'permno', 'jdate' ] )

reversal  = crsp_mom[ [ 'permno', 'jdate', 'ret'] ]
reversal[ 'jdate' ] = reversal[ 'jdate' ] + MonthEnd( 1 )
reversal.rename( columns={ 'ret': 'reversal'}, inplace=True )
crsp_mom3 = pd.merge( crsp_mom2, reversal, how='left', on=['permno', 'jdate'] )

df = df.drop( [ 'date', 'ret', 'retx', 'me' ], axis=1 )
df = pd.merge( crsp_mom3, df, how='left', on=['permno', 'jdate'] )

df = df.groupby( 'permno', as_index=True ).apply( lambda x: x.fillna( method='ffill', limit=11 ) )

df = df[((df['exchcd'] == 1) | (df['exchcd'] == 2) |
                                     (df['exchcd'] == 3)) &
                                    ((df['shrcd'] == 10) | (df['shrcd'] == 11)) ]

df[ 'beme' ] = df[ 'be' ]*1000 / df[ 'lag6_me' ] 
df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

annual_df = df[
    [ 'permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retadj', 'me',
      'lag6_me', 'lag1_me', 'reversal', 'mom', 'gat', 'GP', 'beme'
    ]
]

annual_df[ 'count' ] = annual_df.groupby( ['permno'] ).cumcount()

nyse = annual_df[ ( annual_df['beme']>0 ) & ( annual_df['me']>0 ) & \
             ( annual_df['count']>=1 ) & ( ( annual_df['shrcd']==10 ) | ( annual_df['shrcd']==11 ) )]

nyse_sz = nyse.groupby( ['jdate'] )[ 'me' ].describe( percentiles=[ 0.3, 0.7 ] ).reset_index()
nyse_sz = nyse_sz[ ['jdate','30%','70%'] ].rename( columns={ '30%':'sz30', '70%':'sz70' } )

annual_df = pd.merge( annual_df, nyse_sz, how='inner', on=[ 'jdate'] )

def sz_bucket( row ):
    if row[ 'lag1_me' ]<=row[ 'sz30' ]:
        value = 'Micro'
    elif row[ 'lag1_me' ]<=row[ 'sz70' ]:
        value ='Small'
    elif row[ 'lag1_me' ]>row[ 'sz70' ]:
        value = 'Large'
    else:
        value = ''    
    return value


annual_df['szport'] = np.where( (annual_df['beme']>0) & (annual_df['lag1_me']>0) & (annual_df['count']>=1),
                                 annual_df.apply(sz_bucket, axis=1), '')
annual_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

# winsorized the data 
bm_trim = nyse.groupby( 'jdate' )[ 'beme' ].describe( percentiles=[ .01, .99 ] ).reset_index()
bm_trim = bm_trim[ ['jdate','1%','99%'] ]

me_trim = nyse.groupby( 'jdate' )[ 'lag1_me' ].describe( percentiles=[ .01, .99 ] ).reset_index()
me_trim = me_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'me_1%', '99%': 'me_99%'} )

mom_trim = nyse.groupby( 'jdate' )[ 'mom' ].describe( percentiles=[ .01, .99 ] ).reset_index()
mom_trim = mom_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'mom_1%', '99%': 'mom_99%'})

rev_trim = nyse.groupby( 'jdate' )[ 'reversal' ].describe( percentiles=[ .01, .99 ] ).reset_index()
rev_trim = rev_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'rev_1%', '99%': 'rev_99%'})

gp_trim = nyse.groupby( 'jdate' )[ 'GP' ].describe( percentiles=[ .01, .99 ] ).reset_index()
gp_trim = gp_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'gp_1%', '99%': 'gp_99%'})

ag_trim = nyse.groupby( 'jdate' )[ 'gat' ].describe( percentiles=[ .01, .99 ] ).reset_index()
ag_trim = ag_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'ag_1%', '99%': 'ag_99%'})

annual_df = pd.merge( annual_df, bm_trim, how='left', on='jdate' )
annual_df = pd.merge( annual_df, me_trim, how='left', on='jdate' )
annual_df = pd.merge( annual_df, mom_trim, how='left', on='jdate' )
annual_df = pd.merge( annual_df, rev_trim, how='left', on='jdate' )
annual_df = pd.merge( annual_df, gp_trim, how='left', on='jdate' )
annual_df = pd.merge( annual_df, ag_trim, how='left', on='jdate' )

annual_df[ 'beme' ] = np.where( annual_df[ 'beme' ] >= annual_df['99%'], annual_df['99%'],
                                np.where( annual_df[ 'beme' ] <= annual_df['1%'],  annual_df['1%'],
                                          annual_df[ 'beme' ]) )

annual_df[ 'lag1_me' ] = np.where( annual_df[ 'lag1_me' ] >= annual_df['me_99%'], annual_df['me_99%'],
                              np.where( annual_df[ 'lag1_me' ] <= annual_df['me_1%'],  
                                        annual_df['me_1%'], annual_df[ 'lag1_me' ]) )

annual_df[ 'mom' ] = np.where( annual_df[ 'mom' ] >= annual_df['mom_99%'], annual_df['mom_99%'],
                               np.where( annual_df[ 'mom' ] <= annual_df['mom_1%'], 
                                          annual_df['mom_1%'], annual_df[ 'mom' ]) )

annual_df[ 'reversal' ] = np.where( annual_df[ 'reversal' ] >= annual_df['rev_99%'], annual_df['rev_99%'],
                                np.where( annual_df[ 'reversal' ] <= annual_df['rev_1%'],  
                                         annual_df['rev_1%'], annual_df[ 'reversal' ]) )

annual_df[ 'GP' ] = np.where( annual_df[ 'GP' ] >= annual_df['gp_99%'], annual_df['gp_99%'],
                                np.where( annual_df[ 'GP' ] <= annual_df['gp_1%'],  annual_df['gp_1%'],
                                          annual_df[ 'GP' ]) )

annual_df[ 'gat' ] = np.where( annual_df[ 'gat' ] >= annual_df['ag_99%'], annual_df['ag_99%'],
                                np.where( annual_df[ 'gat' ] <= annual_df['ag_1%'], 
                                          annual_df['ag_1%'], annual_df[ 'gat' ]) )

annual_df = annual_df[ ['permno', 'date', 'jdate', 'shrcd', 'exchcd', 'siccd', 'retadj', 'me', 'beme',
                        'lag6_me', 'lag1_me', 'reversal', 'mom', 'gat', 'GP', 'szport'] ]

annual_df_large = annual_df[ annual_df[ 'szport' ] == 'Large' ]
annual_df_small = annual_df[ annual_df[ 'szport' ] == 'Small' ]
annual_df_micro = annual_df[ annual_df[ 'szport' ] == 'Micro' ]

annual_df_noFinUt = annual_df[ ( ( annual_df['siccd'] > 4900 ) & ( annual_df['siccd'] <= 4949 ) ) |
                               ( ( annual_df['siccd'] > 6000 ) & ( annual_df['siccd'] <= 6799 ) ) ]
annual_df_noFin = annual_df[ ( ( annual_df['siccd'] > 6000 ) & ( annual_df['siccd'] <= 6999 ) ) ]


## FF factors
_ff = db.get_table( library='ff', table='factors_monthly' )
_ff = _ff[ ['date', 'mktrf', 'smb', 'hml', 'rf'] ]
_ff['jdate'] = _ff['date']+MonthEnd( 0 )

## Fama-McBeth regression 
annual_df_noFin[ 'logme' ] = np.log( annual_df_noFin['me'] )
annual_df_noFin[ 'logbeme' ] = np.log( annual_df_noFin['beme'] )
#annual_df_noFin[ np.isinf( annual_df_noFin ) ] = np.nan
annual_df_noFin = pd.merge( annual_df_noFin, _ff[ ['jdate', 'rf'] ],
                            how='left', on='jdate' )
annual_df_noFin[ 'er' ] = annual_df_noFin[ 'retadj' ] - annual_df_noFin[ 'rf' ]
temp = annual_df_noFin#.dropna( axis=0 )
temp.sort_values( by=['permno', 'date'], inplace=True )


temp = temp[ ( temp['jdate']>='1963-07-31') &  (temp['jdate']<='2010-12-31') ].set_index( ['permno', 'jdate' ] )
temp.sort_values( by=['permno', 'date'], inplace=True )

y = temp[ 'er' ]
x = temp[ ['GP', 'logbeme', 'logme', 'reversal', 'mom'] ]

model = FamaMacBeth( y, x )
res = model.fit()

print( res.summary )



