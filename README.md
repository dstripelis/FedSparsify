### 1. Federated Environments

We run the federated experiments for the following data distributions, number of clients, and participation rates:

| Data Distribution | (Number of Clients, Participation Rate) |
| --- |-----------------------------------------| 
| {IID, Non-IID} | (10, 1), (100, 1) |

If participation rate is 1, e.g., (10, 1) then all clients are considered at every federation round, whereas if participation rate is 0.1, e.g., (100, 0.1) then only 10% of the total clients are considered at every round.

IID refers to the case where the training dataset is randomly partitioned across all participating clients and non-IID in the case where the training dataset is first sorted by the prediction class (classification task) and then assigned in a round-robin fashion to the participating clients.

### 2. Learning Domains and Models

| Domain       | Model (#params) | Hyperparameters<sup>1</sup>                             | Centralized Dataset | Federated Dataset                                                                    
|--------------|-----------------|---------------------------------------------------------| --- |--------------------------------------------------------------------------------------|
| FashionMNIST | 2-FC (120k)     | {opt: SGD, lr: 0.02, b: 32, le:4}                       | Train: 60000, Test: 10000 | **IID:** random shuffle, **non-IID(2):** examples from 2 classes per client          |
| CIFAR-10     | 6-CNN (1.6M)    | {opt: SGD w/ Momentum, lr: 0.005, m: 0.75, b: 32, le:4} | Train: 50000, Test: 10000 | **IID:** random shuffle, **non-IID(5):** examples from 5 classes per client          |
| CIFAR-100    | VGG-16 (15M)    | {opt: SGD w/ Momentum, lr: 0.01, m: 0.9, b: 128, le:4}  | Train: 50000, Test: 10000 | **IID:** random shuffle, **non-IID(50):** examples from 50 classes per client        |

[1]: *opt* stands for optimizer, *lr* for learning rate, *b* for batch size, *le* for local epochs within each round.


### 3. How to Run the different Pruning Methods?
The ```fedpurgemerge_main``` directory contains all the source code required to run FedSparsify and other pruning techniques.

Here, we will illustrate a prototypical example based on FashionMNIST using different pruning techniques. The same logic holds for all other domains. For other domains, such as in CIFAR, where more than one models are tested we can uncomment the model generation commands and instead of CNN simply use VGG or ResNet.  

#### 3.1 FedSparsify-Global FashionMNIST
For instance, to run a federated experiments over 100 learners with 10 random learners participating at each round with each learner training for 4 epochs locally and learning a model with a final sparsification degree of 80%, we need to run the ```fedpurgemerge_main/fashion_mnist_main.py``` script as follows:
```
cd fedpurgemerge_main; python3 fashion_mnist_main.py --iid_distribution=True --federation_rounds=200 --learners_num=100 --participation_rate=0.1 --local_epochs=4 --batch_size=32 --train_with_global_mask=True --start_sparsification_at_round=1 --sparsity_level=0.8 --sparsification_frequency=1 --merging_op="FedAvg" --purging_op="fedsparsify-global"
```

#### 3.2 FedSparsify-Local FashionMNIST
Similar flags as the ones used for the above case of FedSparsify-Global, but, we need to change the value of the ```purging_op``` flag to "fedsparsify-local".

#### 3.2 OneShot (with and without FineTuning) FashionMNIST
For OneShot Pruning we can run the following command where we specify when the pruning should take place, e.g., at round 50. This command will simply prune the model at the specified pruning level and keep the sparsity at this degree for the remainder of the federated execution. For instance, if we want to prune a model at 90% soarsity at round 90 and tune the pruned model for another 10 rounds the pruned model, then we simply specify a total number of 100 federation rounds and start model pruning at round 90:

```
cd fedpurgemerge_main; python3 fashion_mnist_main.py --iid_distribution=True --federation_rounds=100 --learners_num=100 --participation_rate=0.1 --local_epochs=4 --batch_size=32 --train_with_global_mask=True --start_sparsification_at_round=90 --sparsity_level=0.8 --merging_op="FedAvg" --purging_op="oneshot"
```

#### 3.3 SNIP/GraSP FashionMNIST
In the ```fedpurgemerge_main/fashion_mnist_main.py``` we need to set the following flags:
```
cd fedpurgemerge_main; python3 fashion_mnist_main.py --iid_distribution=True --federation_rounds=200 --learners_num=100 --participation_rate=0.1 --local_epochs=4 --batch_size=32 --train_with_global_mask=True --sparsity_level=0.8 --merging_op="FedAvg" --purging_op="snip"
```

For GraSP we execute the same command but with ```--purging_op="grasp"```.

#### 3.4 PruneFL FashionMNIST
For PruneFL all the running scripts for all domains and models are located under ```fedpurgemerge_main/prunefl```.


### 4. How to Run the Model Inference Benchmark?
First, we need to convert the tensorflow models into .tflite models and then into the .onnx format. The critical script to accomplish this is located under ```utils/benchmark_model_inference.py```. The script first generates the model structure / architecture and then sets the model weights learned during federated training. Once the new file is generated then the bash script under ```scripts/benchmark_model_inference.sh``` can be used to run the inference benchmark and report the corresponding measurements. 