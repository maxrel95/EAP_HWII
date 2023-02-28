from linearmodels import FamaMacBeth
import pandas as pd
from Utility import fm, fm_iteraction


data = pd.read_csv('Data/benchmark.csv')
data['jdate'] = pd.to_datetime( data['jdate'] )
pd.to_datetime( data['jdate'].unique()).sort_values()
data.set_index( [ 'permno', 'jdate' ], inplace=True )
data[ 'er' ] = data[ 'er' ] *100


fm_b = FamaMacBeth.from_formula( "er ~ 1 + GP+logbm+logme1+reversal+mom", data=data)
res = fm_b.fit(cov_type='kernel')
r2_fm = data.groupby('jdate').apply(fm, var=['GP','logbm','logme1','reversal','mom']).mean()
print(res.summary)

fm2 = FamaMacBeth.from_formula( "er ~ 1 + logbm + logme1 + reversal + mom", data=data)
res2 = fm2.fit(cov_type='kernel')
r2_fm2 = data.groupby('jdate').apply(fm, var=['logbm','logme1','reversal','mom']).mean()
print(res2.summary)

all_df = pd.read_csv( 'Data/all_df.csv' )
all_df['jdate'] = pd.to_datetime( all_df['jdate'] )
all_df.set_index( [ 'permno', 'jdate' ], inplace=True )
all_df[ 'er' ] = all_df[ 'er' ]*100
all_df[ 'residff3' ] = all_df[ 'residff3' ]*100
all_df[ 'residff6' ] = all_df[ 'residff6' ]*100


large_df = all_df[ all_df['szport']=='Large']
small_df = all_df[ all_df['szport']=='Small']
micro_df = all_df[ all_df['szport']=='Micro']

sic_df = all_df[ ( ( all_df['siccd'] < 9000 ) | ( all_df['siccd'] > 9799 ) ) &
                               ( ( all_df['siccd'] < 6000 ) | ( all_df['siccd'] > 6999 ) ) ]

fm_a = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=all_df)
res_a = fm_a.fit( cov_type='kernel' )
r2_fm_a = all_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print( res_a.summary )

fm_ff3 = FamaMacBeth.from_formula( "residff3 ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=all_df)
res_ff3 = fm_ff3.fit(cov_type='kernel')
r2_fm_ff3 = all_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom'], type='residff3').mean()
print( res_ff3.summary )

fm_ff6 = FamaMacBeth.from_formula( "residff6 ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=all_df)
res_ff6 = fm_ff6.fit(cov_type='kernel')
r2_fm_ff6 = all_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom'], type='residff6').mean()
print( res_ff6.summary )

fm_l = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=large_df)
res_l = fm_l.fit(cov_type='kernel')
r2_fm_l = large_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print( res_l.summary )

fm_s = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=small_df)
res_s = fm_s.fit(cov_type='kernel')
r2_fm_s = small_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print( res_s.summary )

fm_m = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=micro_df)
res_m = fm_m.fit(cov_type='kernel')
r2_fm_m = micro_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print( res_m.summary )

fm_ind = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=sic_df)
res_ind = fm_ind.fit(cov_type='kernel')
r2_fm_ind = sic_df.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print( res_ind.summary )

df_q = pd.read_csv( "Data/all_df_q.csv" )
df_q['jdate'] = pd.to_datetime( df_q['jdate'] )
df_q = df_q[ ( df_q['jdate']>='1975-07-30') ]
df_q.set_index( [ 'permno', 'jdate' ], inplace=True )
df_q[ 'er' ] = df_q[ 'er' ]*100

fm_q = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom", data=df_q)
res_q = fm_q.fit(cov_type='kernel')
r2_fm_q = df_q.groupby('jdate').apply(fm, var=['GP', 'gat','logbm','logme1','reversal','mom']).mean()
print(res_q.summary)

# interactions
interaction_df = all_df

fm_mom = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + mom*logbm", data=interaction_df)
res_mom = fm_mom.fit(cov_type='kernel')
r2_fm_mom = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + mom*logbm").mean()
print(res_mom.summary)

fm_bm = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom+ logbm*mom + logbm*logme1", data=interaction_df)
res_bm = fm_bm.fit(cov_type='kernel')
r2_fm_bm = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom+ logbm*mom + logbm*logme1").mean()
print(res_bm.summary)

fm_int = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + logbm*mom + logbm*logme1", data=interaction_df)
res_int = fm_int.fit(cov_type='kernel')
r2_fm_int = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + logbm*mom + logbm*logme1").mean()
print(res_int.summary)

interaction_df = all_df[ (all_df['szport']!='Micro') ]

fm_mom_nomic = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + mom*logbm", data=interaction_df)
res_mom_nomic = fm_mom_nomic.fit(cov_type='kernel')
r2_fm_mom_nomic = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + mom*logbm").mean()
print(res_mom_nomic.summary)

fm_bm_nomic = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom+ logbm*mom + logbm*logme1", data=interaction_df)
res_bm_nomic = fm_bm_nomic.fit(cov_type='kernel')
r2_fm_bm_nomic = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom+ logbm*mom + logbm*logme1").mean()
print(res_bm_nomic.summary)

fm_int_nomic = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + logbm*mom + logbm*logme1", data=interaction_df)
res_int_nomic = fm_int_nomic.fit(cov_type='kernel')
r2_fm_int_nomic = interaction_df.groupby('jdate').apply(fm_iteraction, formula="er ~ 1 + GP+gat+logbm+logme1+reversal+mom + reversal*mom + logbm*mom + logbm*logme1").mean()
print(res_int_nomic.summary)

stats_all = all_df[ ['GP', 'gat', 'logbm', 'logme1', 'reversal', 'mom']].describe( 
    percentiles=[ .01, .25, .5, .75, .99 ]).T.round( 3 ).drop('count', axis=1)
stats_all.round(2).to_latex( 'results/statistics.tex' )

