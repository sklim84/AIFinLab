seed=2024
n_jobs=8

metric_save_path='./_results/results_xgb.csv'

n_estimators=500
gamma=0.0

# hf_ctgan_training
# hf_ctgan_base_100000 hf_ctgan_base_150000 hf_ctgan_base_300000 hf_ctgan_syn_100000 hf_ctgan_syn_150000 hf_ctgan_syn_300000
# hf_copula_base_100000 hf_copula_base_150000 hf_copula_base_300000 hf_copula_syn_100000 hf_copula_syn_150000 hf_copula_syn_300000
# hf_tvae_base_100000 hf_tvae_base_150000 hf_tvae_base_300000 hf_tvae_syn_100000 hf_tvae_syn_150000 hf_tvae_syn_300000

# hf_copula_syn_100000_2 hf_copula_syn_150000_2 hf_copula_syn_300000_2
# hf_ctgan_syn_100000_2 hf_ctgan_syn_150000_2 hf_ctgan_syn_300000_2

# hf_training
# hf_ctgan_base_10000_3 hf_ctgan_base_20000_3 hf_ctgan_base_30000_3 hf_ctgan_syn_10000_3 hf_ctgan_syn_20000_3 hf_ctgan_syn_30000_3
# hf_copula_base_10000_3 hf_copula_base_20000_3 hf_copula_base_30000_3 hf_copula_syn_10000_3 hf_copula_syn_20000_3 hf_copula_syn_30000_3

# KG test
# hf_ctgan_syn_10000KG_10k_1 hf_ctgan_syn_10000KG_10k_000001 hf_ctgan_syn_10000KG_basic_0001 hf_ctgan_syn_10000KG_basic_01 hf_ctgan_base_10000KG_base_1

data_path='../syn_test/results'
origin_data_name=hf_training  # hf_ctgan_training hf_training
for data_name in hf_ctgan_syn_10000KG_10k_1 hf_ctgan_syn_10000KG_10k_000001 hf_ctgan_syn_10000KG_basic_0001 hf_ctgan_syn_10000KG_basic_01 hf_ctgan_base_10000KG_base_1; do
  for learning_rate in 0.01 0.001; do #  0.1 0.01 0.001
    for n_estimators in 100 500; do  # 100 500 1000 1500 2000
      for max_depth in 2 4; do # 4 6 8 10
        for reg_lambda in 0.0 1.0; do # 0.0 0.1 1.0 5.0 10.0
          python -u main_xgb.py \
            --data_path $data_path \
            --seed $seed \
            --n_jobs $n_jobs \
            --metric_save_path $metric_save_path \
            --origin_data_name $origin_data_name \
            --data_name $data_name \
            --n_estimators $n_estimators \
            --learning_rate $learning_rate \
            --max_depth $max_depth \
            --gamma $gamma \
            --reg_lambda $reg_lambda
        done
      done
    done
  done
done

