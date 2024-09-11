# MLL

### A machine learning library (MLL), in Python & Cython

<p align="center">
  <img src="./images/spiral_good2.png" alt="Twin Spiral Data" width="500"/>
  <img src="./images/2class.png" alt="Two Classes" width="500"/>
</p>

This is a machine learning library, made from scratch to challenge myself. No tutorials or previous
code implementation of these ML models were used.

### Example Usage:

```python

from package.Models.Neural_Network import SequentialNeuralNetworkClassifier
from package.Tools import LabelEncoder, split_test_train, map_categorical
from package.data import load_data

# DATA:
xs, ys = load_data()  # e.g. xs has shape (100, 18), and we want to classify the data into 5 different categories
INDPUT_DIM = 18
NUM_CATEGORIES = 5

# Clean/Pre-process
le = LabelEncoder().build(ys)
ys_enc = le.fit_transform(ys)
x_train, x_test, y_train, y_test = split_test_train(xs, ys_enc, 0.25)
ys_train_ohe = map_categorical(y_train, NUM_CATEGORIES)  # Maps i -> [0, ..., 1, ..0] in the i-th index

# Initialise and train model:
snn = SequentialNeuralNetworkClassifier(
	INPUT_DIM,
	NUM_CATEGORIES,
	INTERNAL_LAYERS = [16, 16], 
	EPOCHS = 200,
	learning_rate = 0.01
)
snn.train(x_train, ys_train_ohe)

# Get test results:
result = snn.test(x_test, y_test)
print(f"F1-macro - {result.f1.average_score()}")

```
  

### It implements:  
* Decision Trees - CART (which are highly performant, near sklearns implemenation)  
* Random Forests - using ensembles of these Decision Trees  
* Rotation Forests - An atypical variant of Random Forests (which allow for non-orthogonal decision boundaries)
* Sequential Neural Networks with different activations and erros
(for classification and regression tasks)

* Support Vector Machines (several variations)  
* Logistic Regression  
* Linear Regression
* A suite of useful tools for data cleaning, preparation, and visualisation
* Some toy data sets generators


### Setup:  
Clone the repo:  

	git clone https://github.com/drewdkavi/MLL

Navigate to the repo and install the prerequistes:  
*Note, only NumPy, Cython & SciPy are actually required - the rest are just needed for data-visualisation, and comparisons between this library and sklearn's implemenatations* 

	pip install -r requirements.txt
    
Build Cython files:
  
	python setup.py

Use the library - *to see some demonstrations of the library in action run:*

	python main.py

### Structure:
```sh
.
├── README.md
├── images
│   ├── 2class.png
│   └── spiral_good2.png
├── main.py
├── package
│   ├── Models
│   │   ├── Classifier
│   │   │   ├── DecisionTrees
│   │   │   │   ├── DecisionTreeCython.py
│   │   │   │   ├── generateRule.pyx
│   │   │   │   └── setup.py
│   │   │   ├── LogisticClassification
│   │   │   │   └── LogisticReg.py
│   │   │   ├── Random_Forest
│   │   │   │   ├── RandomForest.py
│   │   │   │   ├── generateRuleRF.pyx
│   │   │   │   └── setup.py
│   │   │   ├── ResultObjects.py
│   │   │   ├── Rotation_Forest
│   │   │   │   ├── RotationForest.py
│   │   │   │   ├── generateRuleRF.pyx
│   │   │   │   └── setup.py
│   │   │   └── SVM
│   │   │       └── Binary_SVM.py
│   │   ├── ModelsTemplate.py
│   │   ├── Neural_Network
│   │   │   ├── SNN.pyx
│   │   │   ├── SNN2.pyx
│   │   │   ├── SequentialNeuralNetworkClassifier.py
│   │   │   ├── SequentialNeuralNetworkRegressor.py
│   │   │   ├── __init__.py
│   │   │   └── setup.py
│   │   ├── NormWrapper.py
│   │   ├── Regressor
│   │   │   └── LinearRegression
│   │   │       └── LeastSquare.py
│   │   └── norm_objects.py
│   ├── Tools
│   │   ├── Extras.py
│   │   ├── LabelEncoder.py
│   │   ├── SplitTestTrain.py
│   │   ├── __init__.py
│   │   └── to_categorical.py
│   ├── data
│   │   ├── Generator.py
│   │   └── __init__.py
│   └── demos
│       ├── bsvm_OVO_demo.py
│       ├── bsvm_OVR_demo.py
│       ├── decisionTree_demo.py
│       ├── llsr_demo.py
│       ├── logreg_demo.py
│       ├── randomForest_demo.py
│       ├── rf_irises.py
│       ├── snn_2class_blobs.py
│       ├── snn_4class_blobs_trial.py
│       ├── snn_breastCancer.py
│       ├── snn_digits.py
│       ├── snn_irises.py
│       └── snn_spiral.py
├── requirements.txt
└── setup.py
```

    
