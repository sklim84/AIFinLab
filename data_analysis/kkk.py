import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 샘플 데이터를 데이터프레임으로 생성
trans = pd.read_csv("C:\\Users\\fuser03\\HF_TRNS_TRAN_CP.csv")
df = pd.DataFrame(trans)

# 데이터프레임의 기본 정보 출력
print("데이터프레임 정보:")
'''
print(df.info())

# 기본 통계량 계산
print("\n기본 통계량:")
print(df.describe())

# 결측치 확인
print("\n결측치 확인:")
print(df.isnull().sum())

# 각 컬럼의 분포 시각화
plt.figure(figsize=(14, 10))
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 3, i)
    if df[column].dtype == 'object':
        df[column].value_counts().plot(kind='bar')
    else:
        sns.histplot(df[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# 상관관계 행렬 계산 및 시각화
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# 개별 변수 간의 상관관계 시각화
sns.pairplot(df)
plt.show()

# 각 컬럼의 유니크 값 확인
print("\n유니크 값 확인:")
for column in df.columns:
    unique_values = df[column].unique()
    print(f"{column}: {unique_values}")

# 거래 날짜별 거래 금액의 변화 추이 시각화
plt.figure(figsize=(10, 6))
## df.groupby('TRAN_DT')['TRAN_AMT'].sum().plot()
df.groupby('tran_dt')['tran_amt'].sum().plot()
plt.title('Transaction Amount Over Time')
plt.xlabel('Transaction Date')
plt.ylabel('Transaction Amount')
plt.show()


# 매체 구분 및 자금 구분에 따른 거래 금액 분포 시각화
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
##sns.boxplot(x='MD_TYPE', y='TRAN_AMT', data=df)
sns.boxplot(x='md_type', y='tran_amt', data=df)
plt.title('Transaction Amount by Media Type')
plt.subplot(1, 2, 2)
##sns.boxplot(x='FND_TYPE', y='TRAN_AMT', data=df)
sns.boxplot(x='fnd_type', y='tran_amt', data=df)
plt.title('Transaction Amount by Fund Type')
plt.tight_layout()
plt.show()



# 금융사기 의심 정보의 분포 시각화
plt.figure(figsize=(6, 4))
## sns.countplot(x='FF_SP_AI', data=df)
sns.countplot(x='ff_sp_ai', data=df)
plt.title('Suspicious Transaction Information')
plt.show()


# 결과를 사용자에게 표시합니다.
import ace_tools as tools; tools.display_dataframe_to_user(name="Financial Transaction Data Analysis", dataframe=df.describe())

# 자금구분 별 금융사기 의심유의정보 값(‘01’, ‘02’, ‘SP’)의 분포 시각화
plt.figure(figsize=(12, 6))
print("start")
sns.boxplot(x='ff_sp_ai', y='fnd_type', data=df)
## sns.countplot(x='FND_TYPE', hue='FF_SP_AI', data=df, order=df['FND_TYPE'].unique(), hue_order=['01', '02', 'SP'])
## sns.countplot(x='fnd_type', hue='ff_sp_ai', data=df, order=df['fnd_type'].unique(), hue_order=['01', '02', 'SP'])
plt.title('Suspicious Transaction Information by Fund Type')
plt.xlabel('ff_sp_ai')
plt.ylabel('fnd_type')
print("end")
plt.show()


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['fnd_type', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)



# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['tran_amt', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['md_type', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['wd_fc_sn', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['dps_fc_sn', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
filtered_df['month'] = filtered_df['tran_dt'].astype(str).str[4:6]

count_table = filtered_df.groupby(['month', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
filtered_df['month'] = filtered_df['tran_dt'].astype(str).str[6:8]

count_table = filtered_df.groupby(['month', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)


# 'ff_sp_ai' 컬럼에 값이 있는 데이터 필터링
filtered_df = df[df['ff_sp_ai'].notnull()]

# 'fnd_type' 별로 'ff_sp_ai' 값의 누적 갯수 계산
count_table = filtered_df.groupby(['tran_tmrg', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 테이블 출력
print("FND_TYPE 별 FF_SP_AI 값의 누적 갯수:")
print(count_table)
'''