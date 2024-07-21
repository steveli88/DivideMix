# DivideMix + Cluster Prior in normal noise
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.2 --lambda_u 0 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.5 --lambda_u 25 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.8 --lambda_u 25 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.9 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.2 --lambda_u 25 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.5 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.8 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.9 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch 300
