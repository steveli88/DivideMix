# DivideMix in Extreme noise
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.92 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch -1
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.95 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch -1
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.98 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch -1
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.92 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch -1
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.95 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch -1

# DivideMix + Cluster Prior in Extreme noise
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.92 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.95 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar10 --data_path ./data/cifar-10-batches-py --noise_mode sym --r 0.98 --lambda_u 50 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.92 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch 300
python Train_cifar.py --dataset cifar100 --data_path ./data/cifar-100-python --noise_mode sym --r 0.95 --lambda_u 150 --p_threshold 0.5 --cluster_prior_epoch 300
