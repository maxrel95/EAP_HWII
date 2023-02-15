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
compustat_annual = db.raw_sql(
    """
                    select gvkey, datadate, at, pstkl, txditc,
                    pstkrv, seq, pstk, gp
                    from compustat_annual.funda
                    where indfmt='INDL' 
                    and datafmt='STD'
                    and popsrc='D'
                    and consol='C'
                    and datadate >= '01/01/1959'
                    """,
                     date_cols=['datadate']
)

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

compustat_annual[ 'year' ] = compustat_annual[ 'datadate' ].dt.year # creat a col with year only 

# create preferrerd stock
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'pstkrv' ].isnull(),
 compustat_annual[ 'pstkl' ], compustat_annual[ 'pstkrv' ] )
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'ps' ].isnull(),compustat_annual[ 'pstk' ],
 compustat_annual['ps'])
compustat_annual[ 'ps' ] = np.where( compustat_annual[ 'ps' ].isnull(), 0, compustat_annual[ 'ps' ])
compustat_annual[ 'txditc' ] = compustat_annual[ 'txditc' ].fillna(0)

# create book equity
compustat_annual[ 'be' ] = compustat_annual['seq']+compustat_annual['txditc']-compustat_annual['ps']
compustat_annual[ 'be' ] = np.where( compustat_annual[ 'be' ]>0, compustat_annual[ 'be' ], np.nan )

# number of years in compustat_annualustat
compustat_annual = compustat_annual.sort_values( by=[ 'gvkey','datadate' ] )
compustat_annual[ 'count' ] = compustat_annual.groupby([ 'gvkey' ] ).cumcount()

