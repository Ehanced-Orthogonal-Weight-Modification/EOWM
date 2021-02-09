### Enhanced Orthogonal Weights Modicifation (EOWM)

This project is the implementation of Defeating Catastrophic Forgetting via Enhanced Orthogonal Weights Modification in Shuffled MNIST and CIFAR-100 setting.

#### Shuffled MNIST

##### Requirements

- python 3.6.12
- torch 0.4.1(PyTorch)
- torchvision 0.2.0
- numpy 1.15.1

#####  Explanations

`run_shuffled_mnist.py` is the main script to train and test our model.

`EOWM.py` is the specific implementation of EOWM.

##### Running

To train and test the model, run the main script `run_shuffled_mnist.py` with command-line arguments.

```python
python run_shuffled_mnist.py 
```

#### CIFAR-100

##### Requirements

- python 3.5.6
- torch 0.3.0(PyTorch)
- torchvision 0.2.0
- numpy 1.15.1

##### Explanations

`cifar_eowm.py` defines the networks' structure.

`eowm.py` is the specific implementation of EOWM. It also contains the specific implementation of training and testing.

`run_cifar100.py` defines the training and testing procedure of our model.

`utils.py` loads CIFAR-100 data and defines functions utilized in `run_cifar.py`.

##### Running

To train and test the model, run the script `run_cifar.py` with command-line arguments.

```python
python run_cifar.py
```

