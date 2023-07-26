#!/bin/bash

modelname='abMIL'
python main.py --model_name $modelname --num_bag_train 50
python main.py --model_name $modelname --num_bag_train 100
python main.py --model_name $modelname --num_bag_train 150
python main.py --model_name $modelname --num_bag_train 200
python main.py --model_name $modelname --num_bag_train 300
python main.py --model_name $modelname --num_bag_train 400
python main.py --model_name $modelname --num_bag_train 500

modelname='sa-abMIL'
python main.py --model_name $modelname --self_att --num_bag_train 50
python main.py --model_name $modelname --self_att --num_bag_train 100
python main.py --model_name $modelname --self_att --num_bag_train 150
python main.py --model_name $modelname --self_att --num_bag_train 200
python main.py --model_name $modelname --self_att --num_bag_train 300
python main.py --model_name $modelname --self_att --num_bag_train 400
python main.py --model_name $modelname --self_att --num_bag_train 500

modelname='gsa-abMIL'
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 50
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 100
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 150
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 200
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 300
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 400
python main.py --model_name gsa-abMIL --self_att --kernel_self_att --num_bag_train 500

