import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

trans = pd.read_csv("C:\\Users\\fuser03\\HF_TRNS_TRAN_CP.csv")

# Don't need for the analysis
analysis1 = trans.drop(columns=['wd_ac_sn', 'dps_ac_sn'])

# sns.countplot(x='wd_fc_sn', data=analysis1)
# plt.title('Withdrawal Companies')
# plt.xticks(rotation=90)
# plt.show()
#
# sns.countplot(x='dps_fc_sn', data=analysis1)
# plt.title('Deposit Companies')
# plt.xticks(rotation=90)
# plt.show()
# analysis1 = pd.get_dummies(analysis1, columns=['wd_fc_sn', 'dps_fc_sn'])
analysis1 = analysis1.drop(columns=['wd_fc_sn', 'dps_fc_sn'])

# Split date
analysis1['month'] = ((analysis1.tran_dt % 10000)//100)
analysis1['month'] = analysis1['month'].apply(lambda x: f"{x:02d}")
analysis1['date'] = (analysis1.tran_dt % 100)
analysis1['date'] = analysis1['date'].apply(lambda x: f"{x:02d}")

# sns.countplot(x='month', data=analysis1)
# plt.title('by month')
# plt.xticks(rotation=90)
# plt.show()
#
# sns.countplot(x='date', data=analysis1, order=sorted(analysis1['date'].unique()))
# plt.title('by date')
# plt.xticks(rotation=90)
# plt.show()
analysis1 = analysis1.drop(columns=['tran_dt'])

# sns.countplot(x='tran_tmrg', data=analysis1, order=sorted(analysis1['tran_tmrg'].unique()))
# plt.title('by time')
# plt.xticks(rotation=90)
# plt.show()
# analysis1 = analysis1.drop(columns=['tran_tmrg'])

# sns.countplot(x='md_type', data=analysis1, order=sorted(analysis1['md_type'].unique()))
# plt.title('by md')
# plt.xticks(rotation=90)
# plt.show()
#
# sns.countplot(x='fnd_type', data=analysis1, order=sorted(analysis1['fnd_type'].unique()))
# plt.title('by fund type')
# plt.xticks(rotation=90)
# plt.show()

# analysis1['ff_sp_ai'].fillna('00')
# sns.countplot(x='ff_sp_ai', data=analysis1)
# plt.title('by fraud suspected')
# plt.xticks(rotation=90)
# plt.show()
# print(analysis1.head())

# # One-hot Encoding
# analysis1 = pd.get_dummies(analysis1, columns=['md_type', 'fnd_type', 'ff_sp_ai'])
analysis1 = pd.get_dummies(analysis1, columns=['ff_sp_ai', 'fnd_type'])
analysis1.drop(columns=['md_type'])

# sns.countplot(x='tran_tmrg', data=trans)
# plt.title('by time')
# plt.xticks(rotation=90)
# plt.show()

# sns.heatmap(trans.values,
#             cbar=False,
#             annot=True,
#             annot_kws={'size' : 20},
#             fmt='.2f',
#             square='True',
#             yticklabels=trans.columns,
#             xticklables=trans.columns)

# 월별 이체건수 : 우상향
# monthly_count = analysis1.groupby('month').size().reset_index(name='count')
# plt.figure(figsize=(10, 6))
# plt.plot(monthly_count['month'].astype(str), monthly_count['count'], marker='o', linestyle='-', color='b')
# for i, row in monthly_count.iterrows():
#     plt.text(str(row['month']), row['count'], str(row['count']), fontsize=12, ha='center', va='bottom')
# plt.xlabel('Month')
# plt.ylabel('Transfer Count')
# plt.title('Monthly Transfer Count')
# plt.grid(True)
# plt.show()

# pivot = analysis1.pivot_table(index='date', columns='month', aggfunc='size', fill_value=0)
# pivot.to_excel('C:\\Users\\fuser03\\daily_count.xlsx')
# print(f'File is created')

corr_matrix = analysis1.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title("Correlation")
plt.show()
