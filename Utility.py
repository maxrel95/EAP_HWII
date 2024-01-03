# Author : Maxime Borel 
# Class : Empirical Asset Pricing 
# Homework II

import numpy as np 
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def ff3model( df ):
    y = df['er']
    if y.shape[0] < 30:
        z = np.empty(y.shape)
        z[:] = np.nan
        temp = pd.DataFrame( z, columns=['residff3'])
        temp = pd.concat( [ df[['permno', 'jdate']], temp], axis=1)
        return temp
    x = df[['mktrf', 'smb', 'hml']]
    x = sm.add_constant( x )
    model = sm.OLS(y.astype('float'), x.astype('float') )
    res = model.fit()
    resid = res.resid + res.params[ 0 ]
    eps =  pd.concat( [ df[['permno', 'jdate']], resid], axis=1)
    eps = eps.rename( columns={ 0: 'residff3' } )
    return eps


def ff6model( df,  ):
    y = df['er']
    if y.shape[0] < 30:
        z = np.empty(y.shape)
        z[:] = np.nan
        temp = pd.DataFrame( z, columns=['residff6'])
        temp = pd.concat( [ df[['permno', 'jdate']], temp], axis=1)
        return temp
    x = df[['mktrf', 'smb', 'hml', 'rmw', 'cma', 'umd']]
    x = sm.add_constant( x )
    model = sm.OLS( y.astype('float'), x.astype('float') )
    res = model.fit()
    resid = res.resid + res.params[ 0 ]
    eps =  pd.concat( [ df[['permno', 'jdate']], resid], axis=1)
    eps = eps.rename( columns={ 0: 'residff6' } )
    return eps


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


def trim( x ): 
    nyse = x[ (x['exchcd']==1) & ( x['beme']>0 ) & ( x['me']>0 ) & \
             ( x['count']>=1 ) & ( ( x['shrcd']==10 ) | ( x['shrcd']==11 ) )]
             
    bm_trim = nyse.groupby( 'jdate' )[ 'logbm' ].describe( percentiles=[ .01, .99 ] ).reset_index()
    bm_trim = bm_trim[ ['jdate','1%','99%'] ]

    me_trim = nyse.groupby( 'jdate' )[ 'logme' ].describe( percentiles=[ .01, .99 ] ).reset_index()
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

    x[ 'logme' ] = np.where( x[ 'logme' ] >= x['me_99%'], x['me_99%'],
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

def aggregate_quarter(series, df):#
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


def fm( df, var, type='er' ):
    x = df[var]
    x = sm.add_constant( x )
    y = df[type]
    mdl = sm.OLS( y,x )
    res = mdl.fit()
    r2 = res.rsquared_adj
    return r2


def fm_iteraction( df, formula ):
    mdl = smf.ols( formula=formula, data=df )
    res = mdl.fit()
    return res.rsquared_adj
