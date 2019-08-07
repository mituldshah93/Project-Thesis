# Project-Thesis

Graph Convolutional Networks - Kipf Enhanced Model
	
Requirements:
	Tensorflow (>0.12)
	networkx

Steps to Follow:

1. Load the dataset to the working folder of the notebook/pycharm - Folder name is "data"

2. Open the "GCN_complete_working model.ipynb" in the notebook

3. Run all the cells for the accuracy and loss calculations

4. Last cell has the visualisation

5. If want to make changes in code:
	
		a. 3 Layered Model : in "GCN(Model)" function, de-comment the layer "Weighted sum" and the input node of WS layer output of first GraphConvolution layer and the input nodes of last GC layer should be output of WS layer

		b. If Clustering co-efficient matrix needs to be implemented, change "def preprocess_adj(adj)" function where "adj_normalized = normalize_adj(adj + diag)"

		c. Change the dataset : in "load_data('cora')", we can set any values of   # 'cora', 'citeseer', 'pubmed' and 'cora_ml'
