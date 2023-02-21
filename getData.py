# Author : Maxime Borel 
# Class : Empirical Asset Pricing 
# Homework II

import numpy as np 
import pandas as pd
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from linearmodels import FamaMacBeth

def trim( x ): 
    nyse = x[ (x['exchcd']==1) & ( x['beme']>0 ) & ( x['me']>0 ) & \
             ( x['count']>=1 ) & ( ( x['shrcd']==10 ) | ( x['shrcd']==11 ) )]
             
    bm_trim = nyse.groupby( 'jdate' )[ 'logbm' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    bm_trim = bm_trim[ ['jdate','1%','99%'] ]

    me_trim = nyse.groupby( 'jdate' )[ 'logme1' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    me_trim = me_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'me_1%', '99%': 'me_99%'} )

    mom_trim = nyse.groupby( 'jdate' )[ 'mom' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    mom_trim = mom_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'mom_1%', '99%': 'mom_99%'})

    rev_trim = nyse.groupby( 'jdate' )[ 'reversal' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    rev_trim = rev_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'rev_1%', '99%': 'rev_99%'})

    gp_trim = nyse.groupby( 'jdate' )[ 'GP' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    gp_trim = gp_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'gp_1%', '99%': 'gp_99%'})

    ag_trim = nyse.groupby( 'jdate' )[ 'gat' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    ag_trim = ag_trim[ ['jdate','1%','99%'] ].rename( columns={ '1%': 'ag_1%', '99%': 'ag_99%'})

    x = pd.merge( x, bm_trim, how='left', on='jdate' )
    x = pd.merge( x, me_trim, how='left', on='jdate' )
    x = pd.merge( x, mom_trim, how='left', on='jdate' )
    x = pd.merge( x, rev_trim, how='left', on='jdate' )
    x = pd.merge( x, gp_trim, how='left', on='jdate' )
    x = pd.merge( x, ag_trim, how='left', on='jdate' )

    x[ 'logbm' ] = np.where( x[ 'logbm' ] >= x['99%'], x['99%'],
                                np.where( x[ 'logbm' ] <= x['1%'],  x['1%'],
                                          x[ 'logbm' ]) )

    x[ 'logme1' ] = np.where( x[ 'logme1' ] >= x['me_99%'], x['me_99%'],
                              np.where( x[ 'logme' ] <= x['me_1%'],  
                                        x['me_1%'], x[ 'logme' ]) )

    x[ 'mom' ] = np.where( x[ 'mom' ] >= x['mom_99%'], x['mom_99%'],
                               np.where( x[ 'mom' ] <= x['mom_1%'], 
                                          x['mom_1%'], x[ 'mom' ]) )

    x[ 'reversal' ] = np.where( x[ 'reversal' ] >= x['rev_99%'], x['rev_99%'],
                                np.where( x[ 'reversal' ] <= x['rev_1%'],  
                                         x['rev_1%'], x[ 'reversal' ]) )

    x[ 'GP' ] = np.where( x[ 'GP' ] >= x['gp_99%'], x['gp_99%'],
                                np.where( x[ 'GP' ] <= x['gp_1%'],  x['gp_1%'],
                                          x[ 'GP' ]) )

    x[ 'gat' ] = np.where( x[ 'gat' ] >= x['ag_99%'], x['ag_99%'],
                                np.where( x[ 'gat' ] <= x['ag_1%'], 
                                          x['ag_1%'], x[ 'gat' ]) )
    return x


db = wrds.Connection( wrds_username = 'maxrel95' )

# Annual fundamental data request
compustat_annual = db.raw_sql("""
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk, gp, revt, cogs
                    from comp.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1960'
                    """, date_cols=['datadate'])

compustat_annual[ 'datadate' ] = compustat_annual[ 'datadate' ] + MonthEnd( 0 )
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
                      where a.date between '01/01/1960' and '12/31/2021'
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

df[ 'txditc' ] = df[ 'txditc' ].fillna( 0 )

# book equity
df[ 'be' ] = df[ 'seq' ] + df[ 'txditc' ] - df[ 'ps' ]
df[ 'be' ] = np.where( df[ 'be' ]>0, df[ 'be' ], np.nan )

# total asset growth
df[ 'at_l1' ] = df.groupby( [ 'gvkey' ] )[ 'at' ].shift( 1 )
df[ 'gat' ] = ( df[ 'at' ] - df[ 'at_l1' ] ) / df[ 'at_l1' ]

# gross profitability
df[ 'GP' ] = ( df[ 'revt' ] - df[ 'cogs' ] ) / df[ 'at' ]#.groupby( [ 'gvkey' ] )[ 'at' ].shift( 1 )

# Momentum 
crsp_mom = db.raw_sql(
                    """
                      select permno, date, ret, retx, shrout, prc
                      from crsp.msf 
                      where date between '01/01/1960' and '12/31/2021'
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
crsp_mom[ 'retadj' ] = ( 1+crsp_mom[ 'ret' ])*(1+crsp_mom[ 'dlret' ] ) - 1

# market cap
crsp_mom[ 'me' ] = ( crsp_mom[ 'prc' ].abs()*crsp_mom[ 'shrout' ] ) /1000
crsp_mom = crsp_mom.sort_values(by=['permno', 'jdate'])

# compute momentum 
mom = crsp_mom[ [ 'permno', 'jdate', 'ret'] ]
mom[ 'gross_ret' ] = 1 + mom[ 'ret' ]
mom[ 'mom' ] = mom.groupby( ['permno'] )[ 'gross_ret' ].rolling( window=11, min_periods=11, closed='left' ).apply( 
                            lambda x: x.prod() ).reset_index( 0, drop=True ) - 1
mom = mom[ ['permno', 'jdate', 'mom'] ].sort_values( by=['permno', 'jdate'] )
crsp_mom1 = pd.merge( crsp_mom, mom, how='left', on=['permno', 'jdate'] )

# compute lagged market cap for bm
me = crsp_mom[ [ 'permno', 'jdate', 'me'] ] 
me[ 'jdate' ] = me[ 'jdate' ] + MonthEnd( 6 )
me.rename( columns={ 'me': 'lag6_me'}, inplace=True )
crsp_mom2 = pd.merge( crsp_mom1, me, how='left', on=[ 'permno', 'jdate' ] )

# compute reversal
crsp_mom2[ 'reversal' ] = crsp_mom2[ 'ret' ]

# merged everythinbg 
df = df.drop( [ 'date', 'ret', 'retx', 'me' ], axis=1 )
df = pd.merge( crsp_mom2, df.drop_duplicates(subset=[ 'permno', 'jdate' ]), how='left', on=['permno', 'jdate'] )

df = df.groupby( 'permno', as_index=True ).apply( lambda x: x.fillna( method='ffill', limit=11 ) )

df = df[((df['exchcd'] == 1) | (df['exchcd'] == 2) | (df['exchcd'] == 3)) &
         ((df['shrcd'] == 10) | (df['shrcd'] == 11)) ]

df[ 'beme' ] = df[ 'be' ] / df[ 'lag6_me' ]
df[ 'logbm' ] = np.log( df[ 'beme' ] )
df[ 'logme' ] = np.log( df[ 'me' ] )
df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

annual_df = df[
    [ 'permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retadj', 'me', 'logme',
      'reversal', 'mom', 'gat', 'GP', 'beme', 'logbm'
    ]
]
annual_df[ 'count' ] = annual_df.groupby( ['permno'] ).cumcount()

# filter dataset
# size
nyse = annual_df[ (annual_df['exchcd']==1) & ( annual_df['beme']>0 ) & ( annual_df['me']>0 ) & \
             ( annual_df['count']>=1 ) & ( ( annual_df['shrcd']==10 ) | ( annual_df['shrcd']==11 ) )]

nyse_sz = nyse.groupby( ['jdate'] )[ 'logme' ].describe( percentiles=[ 0.2, 0.5 ] ).reset_index()
nyse_sz = nyse_sz[ ['jdate','20%','50%'] ].rename( columns={ '20%':'sz20', '50%':'sz50' } )

annual_df = pd.merge( annual_df, nyse_sz, how='inner', on=[ 'jdate'] )

def sz_bucket( row ):
    if row[ 'logme' ]<=row[ 'sz20' ]:
        value = 'Micro'
    elif row[ 'logme' ]<=row[ 'sz50' ]:
        value ='Small'
    elif row[ 'logme' ]>row[ 'sz50' ]:
        value = 'Large'
    else:
        value = ''    
    return value

annual_df['szport'] = np.where( (annual_df['beme']>0) & (annual_df['me']>0) & (annual_df['count']>=1),
                                 annual_df.apply(sz_bucket, axis=1), '')
annual_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

# shift returns
annual_df[ 'retadj_l1' ] = annual_df.groupby( 'permno' )[ 'retadj' ].shift( -1 )
annual_df[ 'logme1' ] = annual_df.groupby( 'permno' )[ 'logme' ].shift( 6 )


# filter on size 
annual_df_large = annual_df[ annual_df[ 'szport' ] == 'Large' ]
annual_df_small = annual_df[ annual_df[ 'szport' ] == 'Small' ]
annual_df_micro = annual_df[ annual_df[ 'szport' ] == 'Micro' ]

# filter on industry 
annual_df_noFinUt = annual_df[ ( ( annual_df['siccd'] < 4900 ) | ( annual_df['siccd'] > 4999 ) ) &
                               ( ( annual_df['siccd'] < 6000 ) | ( annual_df['siccd'] > 6999 ) ) ]
annual_df_noFin = annual_df[ ( ( annual_df['siccd'] < 6000 ) | ( annual_df['siccd'] > 6999 ) ) & (~(annual_df[ 'szport' ] == 'Micro')) ]

# trim data
annual_df_trim = trim( annual_df )

annual_df_large = trim( annual_df_large )
annual_df_small = trim( annual_df_small )
annual_df_micro = trim( annual_df_micro )

annual_df_noFinUt = trim( annual_df_noFinUt )
annual_df_noFin = trim( annual_df_noFin )


temp = annual_df_noFin[ ( annual_df_noFin['jdate']>='1963-07-31') &  (annual_df_noFin['jdate']<='2013-12-31') ]
temp = temp.dropna( subset=['GP', 'logbm', 'logme1', 'reversal', 'mom', 'retadj_l1'] )
temp[ ['GP', 'logbm', 'logme1', 'reversal', 'mom']].describe(percentiles=[ .01, .25, .5, .75, .99 ]).T.round(3)

## FF factors
_ff = db.get_table( library='ff', table='factors_monthly' )
_ff = _ff[ ['date', 'mktrf', 'smb', 'hml', 'rf'] ]
_ff['jdate'] = _ff['date'] + MonthEnd( 0 )
_ff[ 'rfl1' ] = _ff['rf'].shift( -1 ) 

temp = pd.merge( temp, _ff[ ['jdate', 'rfl1'] ], how='left', on='jdate' )
temp[ 'er' ] = temp[ 'retadj_l1' ] - temp[ 'rfl1' ]

temp.sort_values( by=['permno', 'jdate'], inplace=True )
temp.to_csv('test.csv')
temp.set_index( [ 'permno', 'jdate' ], inplace=True )

y = temp[ 'er' ]*100
x = temp[ ['GP', 'logbm', 'logme1', 'reversal', 'mom'] ]

model = FamaMacBeth( y, x )
res = model.fit()

print( res.summary )
