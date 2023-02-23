from linearmodels import FamaMacBeth
import pandas as pd


data = pd.read_csv('Data/benchmark.csv')
data['jdate'] = pd.to_datetime( data['jdate'] )
pd.to_datetime( data['jdate'].unique()).sort_values()
data.set_index( [ 'permno', 'jdate' ], inplace=True )
data[ 'er' ] = data[ 'er' ]*100


fm = FamaMacBeth.from_formula( "er ~ 1 + GP+logbm+logme+reversal+mom", data=data)
res = fm.fit(cov_type='kernel')
print(res.summary)

fm2 = FamaMacBeth.from_formula( "er ~ 1 +logbm+logme+reversal+mom", data=data)
res2 = fm2.fit(cov_type='kernel')
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

fm_a = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=all_df)
res_a = fm_a.fit(cov_type='kernel')
print(res_a.summary)

fm_ff3 = FamaMacBeth.from_formula( "residff3 ~ 1 + GP+gat+logbm+logme+reversal+mom", data=all_df)
res_ff3 = fm_ff3.fit(cov_type='kernel')
print(res_ff3.summary)

fm_ff6 = FamaMacBeth.from_formula( "residff6 ~ 1 + GP+gat+logbm+logme+reversal+mom", data=all_df)
res_ff6 = fm_ff6.fit(cov_type='kernel')
print(res_ff6.summary)

fm_l = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=large_df)
res_l = fm_l.fit(cov_type='kernel')
print(res_l.summary)

fm_s = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=small_df)
res_s = fm_s.fit(cov_type='kernel')
print(res_s.summary)

fm_m = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=micro_df)
res_m = fm_m.fit(cov_type='kernel')
print(res_m.summary)

fm_ind = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=sic_df)
res_ind = fm_ind.fit(cov_type='kernel')
print(res_ind.summary)

df_q = pd.read_csv( "Data/all_df_q.csv" )
df_q['jdate'] = pd.to_datetime( df_q['jdate'] )
df_q = df_q[ ( df_q['jdate']>='1975-07-30') ]
df_q.set_index( [ 'permno', 'jdate' ], inplace=True )
df_q[ 'er' ] = df_q[ 'er' ]*100

fm_q = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom", data=df_q)
res_q = fm_q.fit(cov_type='kernel')
print(res_q.summary)

# interactions
interaction_df = all_df[ (all_df['szport']!='Micro') & ( ( all_df['siccd'] < 6000 ) | ( all_df['siccd'] > 6999 ) )]

fm_mom = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom + reversal*mom + mom*logbm", data=interaction_df)
res_mom = fm_mom.fit(cov_type='kernel')
print(res_mom.summary)

fm_bm = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom+ logbm*mom + logbm*logme", data=interaction_df)
res_bm = fm_bm.fit(cov_type='kernel')
print(res_bm.summary)

fm_int = FamaMacBeth.from_formula( "er ~ 1 + GP+gat+logbm+logme+reversal+mom + reversal*mom + logbm*mom + logbm*logme", data=interaction_df)
res_int = fm_int.fit(cov_type='kernel')
print(res_int.summary)
