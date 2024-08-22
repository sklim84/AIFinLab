seed=2024
gpu='1'

metric_save_path='./_results/results_catb.csv'

iterations=500
boosting_type='Plain'
od_type='Iter'
max_ctr_complexity=4

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
  for learning_rate in 0.1 0.01 0.001 0.0001; do #  0.1 0.01 0.001
    for iterations in 100 500 1000 1500 2000; do  # 100 500 1000 1500 2000
      for depth in 2 4 8 10; do # 4 6 8 10
        for l2_leaf_reg in 0 3 5 7 9 10; do # 1 3 5 7 9
            python -u main_catb.py \
              --data_path $data_path \
              --seed $seed \
              --gpu $gpu \
              --metric_save_path $metric_save_path \
              --origin_data_name $origin_data_name \
              --data_name $data_name \
              --iterations $iterations \
              --learning_rate $learning_rate \
              --depth $depth \
              --l2_leaf_reg $l2_leaf_reg \
              --max_ctr_complexity $max_ctr_complexity \
              --boosting_type $boosting_type \
              --od_type $od_type
        done
      done
    done
  done
done

