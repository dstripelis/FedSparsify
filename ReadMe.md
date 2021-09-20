### Federated Environments

We run the federated experiments for the following data distributions, number of clients, federation rounds,
participation rates and initial states:

| Data Distribution | Number of Clients | Federation Rounds | Participation Rate | Initialization State |
| --- | --- | --- | --- | --- |
| {IID, Non-IID} | {10, 100, 1000} | {500} | {1, 0.5, 0.1} | {random, burnin-consensus, burnin-singleton} |

## Learning Domains and Models

| Domain | Model (#params) | Hyperparameters<sup>1</sup> | Centralized Dataset | Federated Dataset
| --- | --- | --- | --- | --- |
| FashionMNIST | 2-FC () | {opt: SGD, lr: 0.02, b: 32, le:4} | Train: 60000, Test: 10000 | **IID:** random shuffle, **non-IID:** sort by class |
| CIFAR-10 | CNN () | {opt: SGD, lr: 0.005, b: 32, le:4} | Train: 50000, Test: 10000 | **IID:** random shuffle, **non-IID:** sort by class |
| IMDB | LSTM (3M) | {opt: SGD, lr: 0.01, b: 32, le:4} | Train: 25000, Test: 25000 | **IID:** random shuffle, **non-IID:** sort by polarity (class) |

[1]: *opt* stands for optimizer, *lr* for learning rate, *b* for batch size, *le* for local epochs.