import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

csv_data_file = r'test-train-data\test\test_train_data-all.csv'
new_csv_data_file = r'test-train-data\validate\validate_data_new-all.csv'

all_data = pd.concat([pd.read_csv(csv_data_file), pd.read_csv(new_csv_data_file)], axis=0)
all_data_nz = all_data.loc[:, (all_data != 0).any(axis=0)]

mal = all_data_nz[all_data_nz.malware_label == 1]
#mal = pd.DataFrame(pd.DataFrame(MinMaxScaler().fit_transform(mal), columns=mal.columns).mean(), columns=['Malware'])
ben = all_data_nz[all_data_nz.malware_label == 0]
#ben = pd.DataFrame(pd.DataFrame(MinMaxScaler().fit_transform(ben), columns=ben.columns).mean(), columns=['Benign'])

#######################
# CS Number of CS's used
#sns.barplot(data=cs_sum, x=cs_sum.index)
#mal_cs_nz = len(mal_cs_data.loc[:, (mal_cs_data != 0).any(axis=0)].columns)
#ben_cs_nz = len(ben_cs_data.loc[:, (ben_cs_data != 0).any(axis=0)].columns)

# Add labels above bars
#cs_count = pd.DataFrame({'Label': ['Benign Signature Algorithms', 'Malware Signature Algorithms'], 'Count': [ben_cs_nz, mal_cs_nz]})
#ax = sns.barplot(data=cs_count, y='Count', x='Label')
#for bar in ax.patches:
#    ax.annotate(format(bar.get_height(), ''),
#                    (bar.get_x() + bar.get_width() / 2,  
#                    bar.get_height()), ha='center', va='center', 
#                    size=15, xytext=(0, 8), 
#                    textcoords='offset points')

################
# CS top 30 Utilization comparison
mal_cs_data = mal.filter(regex='cs_')
mal_cs_data = mal_cs_data.drop(['cs_len'], axis=1)
mal_cs_data = mal_cs_data.loc[:, (mal_cs_data != 0).any(axis=0)]
mal_cs_data = pd.concat([mal_cs_data, mal.malware_label], axis=1)

ben_cs_data = ben.filter(regex='cs_')
ben_cs_data = ben_cs_data.drop(['cs_len'], axis=1)
ben_cs_data = ben_cs_data.loc[:, (ben_cs_data != 0).any(axis=0)]
ben_cs_data = pd.concat([ben_cs_data, ben.malware_label], axis=1)

#cs_sum = pd.concat([ben_cs_data.sum(), mal_cs_data.sum()], axis=1)
#cs_sum = pd.DataFrame({"Benign": ben_cs_data.sum(), "Malware": mal_cs_data.sum()}, index=ben_cs_data.sum().index)
#cs_sum = pd.DataFrame({"Benign": ben_cs_data.sum().sort_values()}, index=ben_cs_data.sum().sort_values().index)
#cs_sum = pd.DataFrame({"Malware": mal_cs_data.sum().sort_values()}, index=mal_cs_data.sum().sort_values().index)

###################
# Length Fields
#mal_len_data = mal.filter(regex='_len')
#mal_len_data = mal_len_data.loc[:, (mal_len_data != 0).any(axis=0)]

#ben_len_data = ben.filter(regex='_len')
#ben_len_data = ben_len_data.loc[:, (ben_len_data != 0).any(axis=0)]

#cs_sum = pd.concat([pd.DataFrame(ben_len_data.mean(), columns=['Benign']), pd.DataFrame(mal_len_data.mean(), columns=['Malware'])], axis=1)

#################
# Svr features
#mal_svr_data = mal.filter(regex='handshake_')
#mal_svr_data = mal[['dom_in_tranco_1m', 'dom_dga_prob', 'otx_status', 'otx_age', 'urlhaus_status','urlhaus_age']]
#mal_svr_data = mal_svr_data.drop(['svr_tls_ver', 'svr_supported_ver'], axis=1)
#mal_svr_data = pd.DataFrame(pd.DataFrame(MinMaxScaler().fit_transform(mal_svr_data), columns=mal_svr_data.columns).mean(), columns=['Malware'])

#ben_svr_data = ben.filter(regex='handshake_')
#ben_svr_data = ben[['dom_in_tranco_1m', 'dom_dga_prob', 'otx_status', 'otx_age', 'urlhaus_status','urlhaus_age']]
#ben_svr_data = ben_svr_data.drop(['svr_tls_ver', 'svr_supported_ver'], axis=1)
#ben_svr_data = pd.DataFrame(pd.DataFrame(MinMaxScaler().fit_transform(ben_svr_data), columns=ben_svr_data.columns).mean(), columns=['Benign'])

#cs_sum = pd.concat([ben_svr_data, mal_svr_data], axis=1).sort_values(by='Malware')

################
# Ports
#mal_prt = mal[['src_port', 'dst_port']]
#ben_prt = ben[['src_port', 'dst_port']]
#ben_percent = ben_prt.dst_port.unique().shape[0] / ben_prt.dst_port.shape[0]
#mal_percent = mal_prt.dst_port.unique().shape[0] / mal_prt.dst_port.shape[0]
#cs_sum = pd.DataFrame({'Label': ['Benign Unique Dst Ports', 'Malware Unique Dst Ports'], 'Count': [ ben_percent, mal_percent ]})
#cs_sum = pd.DataFrame(ben_prt.dst_port.value_counts()).sort_values(by='dst_port').tail(10)
#cs_sum = cs_sum[cs_sum.index <= 30000]
#cs_sum = pd.concat([ben, mal], axis=1).sort_values(by='Malware').tail(40)
mal_cs_nz = len(mal_cs_data.loc[:, (mal_cs_data != 0).any(axis=0)].columns)
ben_cs_nz = len(ben_cs_data.loc[:, (ben_cs_data != 0).any(axis=0)].columns)

# Add labels above bars
cs_count = pd.DataFrame({'Label': ['Benign Cipher Suites', 'Malware Cipher Suites'], 'Count': [ben_cs_nz, mal_cs_nz]})

ax = sns.barplot(data=cs_count, y='Count', x='Label', orient='v')

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for bar in ax.patches:
    ax.annotate(format(bar.get_height(), ''),
                    (bar.get_x() + bar.get_width() / 2,  
                    bar.get_height()), ha='center', va='center', 
                    size=8, xytext=(0, 8), 
                    textcoords='offset points')
ax.set(xlabel='Label', ylabel='Count')
#cs_sum.plot.barh(rot=0)
plt.show()