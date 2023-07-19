source activate pytorch-py39
python test.py --model hierarchical --dataset_mode BJAir --pred_attr NO2_Concentration --gpu_ids 2 --config config1 --file_time 20230111T201534 --epoch best