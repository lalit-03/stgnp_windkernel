source activate pytorch-py39
python test.py --model hierarchical --dataset_mode BJAir --pred_attr PM25_Concentration --gpu_ids 0 --config config1 --file_time 20230706T034043 --epoch best