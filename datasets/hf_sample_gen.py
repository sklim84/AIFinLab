import pandas as pd
import random
import datetime


# 샘플 데이터 생성
def generate_sample_data(num_samples=100):

    data = []

    # 자주 사용되는 출금 계좌와 입금 계좌 쌍 생성
    common_nodes = [(random.randint(100, 999), random.randint(1000000000000000, 9999999999999999)) for _ in range(10)]

    for _ in range(num_samples):
        TRAN_DT = (datetime.date.today() - datetime.timedelta(days=random.randint(0, 365))).strftime('%Y%m%d')
        # TRAN_TMRG = f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}{random.randint(0, 59):02d}"
        TRAN_TMRG =  random.choice(['00', '03', '06', '08', '12', '15', '18', '21'])

        # WD_FC_SN = random.randint(100, 999)
        # WD_AC_SN = random.randint(1000000000000000, 9999999999999999)
        if random.random() < 0.5:
            WD_FC_SN, WD_AC_SN = random.choice(common_nodes)
        else:
            WD_FC_SN = random.randint(100, 999)
            WD_AC_SN = random.randint(1000000000000000, 9999999999999999)

        # DPS_FC_SN = random.randint(100, 999)
        # DPS_AC_SN = random.randint(1000000000000000, 9999999999999999)
        if random.random() < 0.5:
            DPS_FC_SN, DPS_AC_SN = random.choice(common_nodes)
        else:
            DPS_FC_SN = random.randint(100, 999)
            DPS_AC_SN = random.randint(1000000000000000, 9999999999999999)

        TRAN_AMT = random.randint(1000, 1000000)
        MD_TYPE = random.choice(['01', '02', '03', '04', '05', '06', '07', '08'])
        FND_TYPE = random.choice(
            ['00', '01', '02', '03', '04', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
             '24', '25', '26', '27', '28', '30'])
        FF_SP_AI = random.choice(['00', '01', '02', 'SP'])

        data.append(
            [TRAN_DT, TRAN_TMRG, WD_FC_SN, WD_AC_SN, DPS_FC_SN, DPS_AC_SN, TRAN_AMT, MD_TYPE, FND_TYPE, FF_SP_AI])

    columns = ['TRAN_DT', 'TRAN_TMRG', 'WD_FC_SN', 'WD_AC_SN', 'DPS_FC_SN', 'DPS_AC_SN', 'TRAN_AMT', 'MD_TYPE',
               'FND_TYPE', 'FF_SP_AI']
    df = pd.DataFrame(data, columns=columns)
    return df

n_samples = 100
sample_data = generate_sample_data(n_samples)
print(sample_data.head())

sample_data.to_csv(f'./hf_sample_{n_samples}.csv', index=False)
