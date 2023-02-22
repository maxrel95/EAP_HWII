# Author : Maxime Borel 
# Class : Empirical Asset Pricing 
# Homework II

import numpy as np 
import pandas as pd
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from Utilitiy import *


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

me4 = crsp_mom[ [ 'permno', 'jdate', 'me'] ] 
me4[ 'jdate' ] = me4[ 'jdate' ] + MonthEnd( 4 )
me4.rename( columns={ 'me': 'lag4_me'}, inplace=True )
crsp_mom2 = pd.merge( crsp_mom2, me4, how='left', on=[ 'permno', 'jdate' ] )

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

annual_df['szport'] = np.where( (annual_df['beme']>0) & (annual_df['me']>0) & (annual_df['count']>=1),
                                 annual_df.apply(sz_bucket, axis=1), '')
annual_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

# shift returns
annual_df[ 'retadj_l1' ] = annual_df.groupby( 'permno' )[ 'retadj' ].shift( -1 )
annual_df[ 'logme1' ] = annual_df.groupby( 'permno' )[ 'logme' ].shift( 6 )

# filter date
annual_df = annual_df[ ( annual_df['jdate']>='1963-07-31') ]

# filter on size 
annual_df_large = annual_df[ annual_df[ 'szport' ] == 'Large' ]
annual_df_small = annual_df[ annual_df[ 'szport' ] == 'Small' ]
annual_df_micro = annual_df[ annual_df[ 'szport' ] == 'Micro' ]

# filter on industry 
annual_df_noFinUt = annual_df[ ( ( annual_df['siccd'] < 4900 ) | ( annual_df['siccd'] > 4999 ) ) &
                               ( ( annual_df['siccd'] < 6000 ) | ( annual_df['siccd'] > 6999 ) ) ]
annual_df_noFin = annual_df[ ( ( annual_df['siccd'] < 6000 ) | ( annual_df['siccd'] > 6999 ) ) &
                             (~(annual_df[ 'szport' ] == 'Micro')) ]

# trim data
annual_df_trim = trim( annual_df )

annual_df_large_trim = trim( annual_df_large )
annual_df_small_trim = trim( annual_df_small )
annual_df_micro_trim = trim( annual_df_micro )

annual_df_noFinUt_trim = trim( annual_df_noFinUt )
annual_df_noFin_trim = trim( annual_df_noFin )

## FF factors
_ff = db.get_table( library='ff', table='fivefactors_monthly' )
_ff = _ff[ ['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd', 'rf'] ]
_ff['jdate'] = _ff['date'] + MonthEnd( 0 )
_ff[ 'rfl1' ] = _ff['rf'].shift( -1 ) 
_ff[ ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd'] ] = _ff[ ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd'] ].shift( -1 ) 

# benchmark
benchmark = annual_df_noFin_trim[ ( annual_df_noFin_trim['jdate']>='1963-07-31') &  (annual_df_noFin_trim['jdate']<='2013-12-31') ]
benchmark = benchmark.dropna( subset=['GP', 'logbm', 'logme1', 'reversal', 'mom', 'retadj_l1'] )
annual_df_noFin.dropna( subset=['GP', 'logbm', 'logme1', 'reversal', 'mom', 'retadj_l1'] )[ ['GP', 'logbm',
 'logme1', 'reversal', 'mom']].describe(percentiles=[ .01, .25, .5, .75, .99 ]).T.round(3)
benchmark = pd.merge( benchmark, _ff, how='left', on='jdate' )
benchmark[ 'er' ] = benchmark[ 'retadj_l1' ] - benchmark[ 'rfl1' ]
aa = benchmark.groupby( 'permno', as_index=False ).apply( ff3model )
bb = benchmark.groupby( 'permno', as_index=False ).apply( ff6model )
benchmark = pd.merge( benchmark, aa, how='inner', on=['permno', 'jdate'] )
benchmark = pd.merge( benchmark, bb, how='inner', on=['permno', 'jdate'] )
benchmark.sort_values( by=['permno', 'jdate'], inplace=True )
benchmark.to_csv('Data/benchmark23.csv')

# all data
stats_df = annual_df.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )

stats_all = stats_df[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom']].describe( 
    percentiles=[ .01, .25, .5, .75, .99 ]).T.round( 3 ).drop('count', axis=1)
corr_df_pearson = stats_df[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom'] ].corr().round( 3 )
corr_df_spearman = stats_df[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom'] ].corr( method='spearman' ).round( 3 )
stats_all.to_latex( 'results/statistics.tex' )
corr_df_pearson.to_latex( 'results/pearsoncorr.tex' )
corr_df_spearman.to_latex( 'results/spearmancorr.tex' )

# export
all_df = annual_df_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
all_df = pd.merge( all_df, _ff, how='left', on='jdate' )
all_df[ 'er' ] = all_df[ 'retadj_l1' ] - all_df[ 'rfl1' ]
aa = all_df.groupby( 'permno', as_index=False ).apply( ff3model )
bb = all_df.groupby( 'permno', as_index=False ).apply( ff6model )
all_df = pd.merge( all_df, aa, how='inner', on=['permno', 'jdate'] )
all_df = pd.merge( all_df, bb, how='inner', on=['permno', 'jdate'] )
all_df.sort_values( by=['permno', 'jdate'], inplace=True )
all_df.to_csv( 'Data/all_df.csv' )

# Large
large_df = annual_df_large_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
large_df = pd.merge( large_df, _ff, how='left', on='jdate' )
large_df[ 'er' ] = large_df[ 'retadj_l1' ] - large_df[ 'rfl1' ]
aa = large_df.groupby( 'permno', as_index=False ).apply( ff3model )
bb = large_df.groupby( 'permno', as_index=False ).apply( ff6model )
large_df = pd.merge( large_df, aa, how='inner', on=['permno', 'jdate'] )
large_df = pd.merge( large_df, bb, how='inner', on=['permno', 'jdate'] )
large_df.sort_values( by=['permno', 'jdate'], inplace=True )
large_df.to_csv( 'Data/large_df.csv' )

# Small
small_df = annual_df_small_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
small_df = pd.merge( small_df, _ff, how='left', on='jdate' )
small_df[ 'er' ] = small_df[ 'retadj_l1' ] - small_df[ 'rfl1' ]
aa = small_df.groupby( 'permno', as_index=False ).apply( ff3model )
bb = small_df.groupby( 'permno', as_index=False ).apply( ff6model )
small_df = pd.merge( small_df, aa, how='inner', on=['permno', 'jdate'] )
small_df = pd.merge( small_df, bb, how='inner', on=['permno', 'jdate'] )
small_df.sort_values( by=['permno', 'jdate'], inplace=True )
small_df.to_csv( 'Data/small_df.csv' )

# micro
micro_df = annual_df_micro_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
micro_df = pd.merge( micro_df, _ff, how='left', on='jdate' )
micro_df[ 'er' ] = micro_df[ 'retadj_l1' ] - micro_df[ 'rfl1' ]
aa = micro_df.groupby( 'permno', as_index=False ).apply( ff3model )
bb = micro_df.groupby( 'permno', as_index=False ).apply( ff6model )
micro_df = pd.merge( micro_df, aa, how='inner', on=['permno', 'jdate'] )
micro_df = pd.merge( micro_df, bb, how='inner', on=['permno', 'jdate'] )
micro_df.sort_values( by=['permno', 'jdate'], inplace=True )
micro_df.to_csv( 'Data/micro_df.csv' )

# exclude Fin/Util
noFinUtil_df = annual_df_noFinUt_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
noFinUtil_df = pd.merge( noFinUtil_df, _ff, how='left', on='jdate' )
noFinUtil_df[ 'er' ] = noFinUtil_df[ 'retadj_l1' ] - noFinUtil_df[ 'rfl1' ]
aa = noFinUtil_df.groupby( 'permno', as_index=False ).apply( ff3model )
bb = noFinUtil_df.groupby( 'permno', as_index=False ).apply( ff6model )
noFinUtil_df = pd.merge( noFinUtil_df, aa, how='inner', on=['permno', 'jdate'] )
noFinUtil_df = pd.merge( noFinUtil_df, bb, how='inner', on=['permno', 'jdate'] )
noFinUtil_df.sort_values( by=['permno', 'jdate'], inplace=True )
noFinUtil_df.to_csv( 'Data/noFinUtil_df.csv' )


####################################################################
####################################################################
########################## Quarterly data ##########################
####################################################################
####################################################################

compustat_q = db.raw_sql("""
                    select gvkey, datadate, atq, pstkq, txditcq,
                    seqq, revtq, cogsq
                    from comp.fundq
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1960'
                    """, date_cols=['datadate'])

compustat_q[ 'datadate' ] = compustat_q[ 'datadate' ] + MonthEnd( 0 )
compustat_q = compustat_q.sort_values(by=['gvkey', 'datadate']).drop_duplicates()
compustat_q['atq'] = np.where(compustat_q['atq'] == 0, np.nan, compustat_q['atq'])

ccm1 = pd.merge(compustat_q, link_table, how='left', on=['gvkey'])
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['datadate'] + MonthEnd(4)  # we change quarterly lag here

ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]

data_rawq = pd.merge(crsp2, ccm2, how='inner', on=['permno', 'jdate'])

data_rawq = data_rawq[((data_rawq['exchcd'] == 1) | (data_rawq['exchcd'] == 2) | (data_rawq['exchcd'] == 3)) &
                   ((data_rawq['shrcd'] == 10) | (data_rawq['shrcd'] == 11))]

data_rawq['me'] = data_rawq['me']/1000  # CRSP ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
data_rawq['me'] = np.where(data_rawq['me'] == 0, np.nan, data_rawq['me'])
data_rawq = data_rawq.dropna(subset=['me'])

# deal with the duplicates
data_rawq.loc[data_rawq.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()]
data_rawq.loc[data_rawq.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()]

data_rawq = data_rawq.sort_values(by=['permno', 'jdate'])

data_rawq['beq'] = np.where(data_rawq['seqq']>0, data_rawq['seqq']+data_rawq['txditcq']-data_rawq['pstkq'], np.nan)
data_rawq['beq'] = np.where(data_rawq['beq']<=0, np.nan, data_rawq['beq'])

data_rawq['atq_l4'] = data_rawq.groupby(['permno'])['atq'].shift(4)
data_rawq['gat'] = (data_rawq['atq']-data_rawq['atq_l4'])/data_rawq['atq_l4']

data_rawq['revtq4'] = aggregate_quarter( 'revtq', data_rawq )
data_rawq['cogsq4'] = aggregate_quarter( 'cogsq', data_rawq )
data_rawq['GP'] = (data_rawq['revtq']-data_rawq['cogsq'])/data_rawq['atq_l4']

data_rawq = data_rawq.drop( [ 'date', 'ret', 'retx', 'me' ], axis=1 )
data_rawq = pd.merge( crsp_mom2, data_rawq.drop_duplicates(subset=[ 'permno', 'jdate' ]), how='left', on=['permno', 'jdate'] )

data_rawq = data_rawq.groupby( 'permno', as_index=True ).apply( lambda x: x.fillna( method='ffill', limit=2 ) )

data_rawq = data_rawq[((data_rawq['exchcd'] == 1) | (data_rawq['exchcd'] == 2) | (data_rawq['exchcd'] == 3)) &
         ((data_rawq['shrcd'] == 10) | (data_rawq['shrcd'] == 11)) ]

data_rawq[ 'beme' ] = data_rawq[ 'beq' ] / data_rawq[ 'lag4_me' ]
data_rawq[ 'logbm' ] = np.log( data_rawq[ 'beme' ] )
data_rawq[ 'logme' ] = np.log( data_rawq[ 'me' ] )
data_rawq.sort_values( by=[ 'permno', 'jdate' ], inplace=True )

quarterly_df = data_rawq[
    [ 'permno', 'date', 'jdate', 'datadate', 'shrcd', 'exchcd', 'siccd', 'retadj', 'me', 'logme',
      'reversal', 'mom', 'gat', 'GP', 'beme', 'logbm'
    ]
]
quarterly_df[ 'count' ] = quarterly_df.groupby( ['permno'] ).cumcount()

# filter dataset
# size
nyse_q = quarterly_df[ (quarterly_df['exchcd']==1) & ( quarterly_df['beme']>0 ) & ( quarterly_df['me']>0 ) & \
             ( quarterly_df['count']>=1 ) & ( ( quarterly_df['shrcd']==10 ) | ( quarterly_df['shrcd']==11 ) )]

nyse_sz_q = nyse_q.groupby( ['jdate'] )[ 'logme' ].describe( percentiles=[ 0.2, 0.5 ] ).reset_index()
nyse_sz_q = nyse_sz_q[ ['jdate','20%','50%'] ].rename( columns={ '20%':'sz20', '50%':'sz50' } )

quarterly_df = pd.merge( quarterly_df, nyse_sz_q, how='inner', on=[ 'jdate'] )

quarterly_df['szport'] = np.where( (quarterly_df['beme']>0) & (quarterly_df['me']>0) & (quarterly_df['count']>=1),
                                 quarterly_df.apply(sz_bucket, axis=1), '')
quarterly_df.sort_values( by=[ 'permno', 'jdate' ], inplace=True )


quarterly_df[ 'retadj_l1' ] = quarterly_df.groupby( 'permno' )[ 'retadj' ].shift( -1 )
quarterly_df = quarterly_df[ ( quarterly_df['jdate']>='1963-07-31') ]

quarterly_df_noFin = quarterly_df[ ( ( quarterly_df['siccd'] < 6000 ) | ( quarterly_df['siccd'] > 6999 ) ) &
                             (~(quarterly_df[ 'szport' ] == 'Micro')) ]

quarterly_df_trim = trim( quarterly_df )
quarterly_df_noFin_trim = trim( quarterly_df_noFin )

# benchmark
benchmark_q = quarterly_df_noFin_trim[ ( quarterly_df_noFin_trim['jdate']>='1963-07-31') &  (quarterly_df_noFin_trim['jdate']<='2013-12-31') ]
benchmark_q = benchmark_q.dropna( subset=['GP', 'logbm', 'reversal', 'mom', 'retadj_l1'] )
quarterly_df_noFin_trim.dropna( subset=['GP', 'logbm', 'reversal', 'mom', 'retadj_l1'] )[ ['GP', 'logbm',
             'reversal', 'mom']].describe(percentiles=[ .01, .25, .5, .75, .99 ]).T.round(3)
benchmark_q = pd.merge( benchmark_q, _ff, how='left', on='jdate' )
benchmark_q[ 'er' ] = benchmark_q[ 'retadj_l1' ] - benchmark_q[ 'rfl1' ]
aa = benchmark_q.groupby( 'permno', as_index=False ).apply( ff3model )
bb = benchmark_q.groupby( 'permno', as_index=False ).apply( ff6model )
benchmark_q = pd.merge( benchmark_q, aa, how='inner', on=['permno', 'jdate'] )
benchmark_q = pd.merge( benchmark_q, bb, how='inner', on=['permno', 'jdate'] )
benchmark_q.sort_values( by=['permno', 'jdate'], inplace=True )
benchmark_q.to_csv('Data/benchmark_q.csv')

# all data
stats_df_q = quarterly_df_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )

stats_all_q = stats_df_q[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom']].describe( 
    percentiles=[ .01, .25, .5, .75, .99 ]).T.round( 3 ).drop('count', axis=1)
corr_df_pearson_q = stats_df_q[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom'] ].corr().round( 3 )
corr_df_spearman_q = stats_df_q[ ['GP', 'gat', 'logbm', 'logme', 'reversal', 'mom'] ].corr( method='spearman' ).round( 3 )
stats_all_q.to_latex( 'results/statistics_q.tex' )
corr_df_pearson_q.to_latex( 'results/pearsoncorr_q.tex' )
corr_df_spearman_q.to_latex( 'results/spearmancorr_q.tex' )

# export
all_df_q = quarterly_df_trim.dropna( subset=[ 'GP', 'gat', 'logbm', 'logme', 'reversal', 'mom', 'retadj_l1' ] )
all_df_q = pd.merge( all_df_q, _ff, how='left', on='jdate' )
all_df_q[ 'er' ] = all_df_q[ 'retadj_l1' ] - all_df_q[ 'rfl1' ]
aa = all_df_q.groupby( 'permno', as_index=False ).apply( ff3model )
bb = all_df_q.groupby( 'permno', as_index=False ).apply( ff6model )
all_df_q = pd.merge( all_df_q, aa, how='inner', on=['permno', 'jdate'] )
all_df_q = pd.merge( all_df_q, bb, how='inner', on=['permno', 'jdate'] )
all_df_q.sort_values( by=['permno', 'jdate'], inplace=True )
all_df_q.to_csv( 'Data/all_df_q.csv' )

