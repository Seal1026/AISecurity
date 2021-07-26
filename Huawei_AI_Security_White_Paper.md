### Denfence of poisoning attack

#### 1. Filtering of traning data
Detect and purify malicious input data in training data based on its label.
#### 2. Regression analysis
Noise and abnormality detection implemented through defining approriate loss functions or data characteristics distribution.
#### 3. Ensemble analysis
Composited of multiple sub-systems, the deicision of overall system is determined through voting of them, which enhanced its resistance to malicious attacks.

### Denfence of backdoor attack
#### 1. Input pre-processing
Filter out input that could trigger backdoors which would interfere model results.
#### 2. Model pruning
Prune the neurons and keep model's functionalty so that possibility of having attack-hidden neurons will be reduced.

### Denfence of model/data stealing
#### 1.Private Aggregation of Teacher Ensembles (PATE)
Split training data into multiple subsets, then eachare usd to train an individual DNN model. These models are then used to train a student model bu voting, which gurantee the privacy of training data.
#### 2. Differentially private protection
Add noise to data or models by means of differential privacy
in the model training stage.
#### 3. Model watermarking
Special recognization neuron in model enables special input that checks its authentity.

### Model
1. Detectablity of malicious input or output mislead by attacks.
2. Verifiability of undamental working principle
3. Explainability of decision making

### Reminder: Matric
left all 0, right all 1, we want to make a classification. When boundary is more left, our classification is bold, 
high False positive right; more right, our classification is more conversative, 
less False positive rate, high false negivative rate.
