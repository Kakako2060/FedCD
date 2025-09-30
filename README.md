# Two-Stage Heterogeneous Federated Learning Algorithm with Theoretical Guarantees

Research code that accompanies the paper [Two-Stage Heterogeneous Federated Learning Algorithm with Theoretical Guarantees].
It contains implementation of the following algorithms:
* **FedCD** (the proposed algorithm) ([code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverpFedGen.py)).


## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Cifar10** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 99% of the total available training samples:
<pre><code>cd FedCD/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.99 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedCD/data/Cifar/u20c10-alpha0.1-ratio0.99/
</code></pre>
    

- Similarly, to generate *non-iid* **Cifar100** Dataset, using 99% of the total available training samples:
<pre><code>cd FedGen/data/Cifa100
python generate_niid_dirichlet.py --n_class 100 --sampling_ratio 0.99 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FedGen/data/Cifa100/u20c100-alpha0.1-ratio0.99/
</code></pre> 

## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the *Cifar10* Dataset:
```
python main.py --dataset Cifar-alpha0.1-ratio0.5 --algorithm FedCD --batch_size 32 --num_glob_iters 200 --local_epochs 5 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 5 
python main.py --dataset Cifar-alpha0.1-ratio0.5 --algorithm FedCD --batch_size 32 --num_glob_iters 200 --local_epochs 5 --num_users 10 --lamda 1 --learning_rate 0.01 --model resnet --personal_learning_rate 0.01 --times 5 

```
----

##### Run experiments on the *Cifar100* Dataset:
```
python main.py --dataset Cifa100-alpha0.1-ratio0.99 --algorithm FedCD --batch_size 16 --num_glob_iters 200 --local_epochs 5 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 5 
python main.py --dataset Cifa100-alpha0.1-ratio0.99 --algorithm FedCD --batch_size 32 --num_glob_iters 200 --local_epochs 5 --num_users 10 --lamda 1 --learning_rate 0.01 --model resnet --personal_learning_rate 0.01 --times 5 

```
----

##### Run experiments on the *COVID-FL* Dataset:
```
python main.py --dataset COVID-alpha0-ratio1.0 --algorithm FedCD --batch_size 32 --num_glob_iters 200 --local_epochs 5 --num_users 10 --lamda 1 --learning_rate 0.01 --model resnet --personal_learning_rate 0.01 --times 5 

```
----
##### Run experiments on the *Retina* Dataset:
```
python main.py --dataset Retina-alpha0-ratio1.0 --algorithm FedCD --batch_size 32 --num_glob_iters 200 --local_epochs 5 --num_users 5 --lamda 1 --learning_rate 0.01 --model resnet --personal_learning_rate 0.01 --times 5 

```
----
### Plot
For the input attribute **algorithms**, list the name of algorithms and separate them by comma, e.g. `--algorithms FedCD`
```
  python main_plot.py --dataset Cifar-alpha0.1-ratio0.99 --algorithms FedCD --batch_size 16 --local_epochs 5 --num_users 10 --num_glob_iters 200 --plot_legend 1
```
