ConfGCN Model:
	
Requirements:
  Tensorflow (>1.x)
  Python 3.x

Steps to Follow:
  1. Load the dataset to the working folder of the notebook/pycharm - Folder name is "data"
  2. Open the "Conf_GCN working model.ipynb" in the notebook
  3. Run all the cells for the accuracy and loss calculations
  4. If want to make changes in code:
  
			a. 3 Layered Model : in "add_model(self)" function, de-comment the layer "Weighted sum" and the input node of WS layer should be same as output of first GraphConvolution layer and the input nodes of last GC layer should be similar to output of WS layer

			b. If Clustering co-efficient matrix needs to be implemented, change "def preprocess_adj(adj)" function where "adj_normalized = normalize_adj(adj + diag)"

			c. Change the dataset : in "parser.add_argument('-data',  default='cora',help='Dataset to use')"  we can set any values of   # 'cora', 'citeseer', 'pubmed' and 'cora_ml'

Best Combinations of models with all the datasets are:

•	Pubmed: 

	o	Hidden Layers : 1
	o	Number of Nodes in Hidden Layer : 16
	o	Loss : Cross Entropy Softmax V2
	o	Activation : Relu6 / Selu
	o	Optimizer : ADAM
	o	Matrix : Adjacency + Identity
	o	Learning Rate & Epochs : 0.001 & 250

•	Cora:

	o	Hidden Layers : 1
	o	Number of Nodes in Hidden Layer : 32
	o	Loss : Cross Entropy Softmax V2
	o	Activation : Relu6 / Selu
	o	Optimizer : ADAM
	o	Matrix : Adjacency + Clustering Co-efficient Matrix
	o	Learning Rate & Epochs : 0.001 & 250

•	Citeseer: 

	o	Hidden Layers : 1
	o	Number of Nodes in Hidden Layer : 16 or 32 (Both are providing Better results)
	o	Loss : Cross Entropy Softmax V2
	o	Activation : Relu6 / Selu
	o	Optimizer : ADAM
	o	Matrix : Adjacency + Identity
	o	Learning Rate & Epochs : 0.001 & 250



•	CoraML:

	o	Hidden Layers : 1, but with Multiple Inputs
	o	Number of Nodes in Hidden Layer : 16 with output layer is accepting the input as summation of both Input Layer’s outputs and Canonical form having weights as 0.3 for layer having ‘Relu’ activation and 0.7 for layer having ‘Relu6 / Selu’ activation function
	o	Loss : Cross Entropy Softmax V2
	o	Activation : Relu, Relu6 / Selu
	o	Optimizer : ADAM
	o	Matrix : Adjacency + Identity
	o	Learning Rate & Epochs : 0.001 & 250
