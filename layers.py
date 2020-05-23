import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes, activation):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		self.activation = activation
		# Stores the outgoing summation of weights * feautres 
		self.data = None
		# np.random.seed(42)

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		if self.activation == 'relu':
			temp=X@self.weights+self.biases
			# print("temp",temp)
			return relu_of_X(temp)

			raise NotImplementedError
		elif self.activation == 'softmax':
			temp=X@self.weights+self.biases
			return softmax_of_X(temp)
			raise NotImplementedError
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################
		
	def backwardpass(self, lr, activation_next,activation_curr, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_curr.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		if self.activation == 'relu':
			inp_delta = gradient_relu_of_X(activation_next, delta,1)*delta
			# print("activation_next",activation_next)
			temp=inp_delta@((self.weights).T)
			del_w=lr * (activation_curr).T @ inp_delta
			# print("del_w",del_w)
			self.weights=self.weights-del_w
			self.biases=self.biases-lr* (np.ones((1,n))@ inp_delta)
			return temp
		elif self.activation == 'softmax':
			inp_delta =gradient_softmax_of_X(activation_next, delta)
			temp=inp_delta@((self.weights).T)
			del_w=lr * (activation_curr).T @ inp_delta
			self.weights=self.weights-del_w
			self.biases=self.biases-lr* (np.ones((1,n))@ inp_delta)
			return temp
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###########################	####################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride, activation):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride
		self.activation = activation
		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]
		# final_col=int((self.in_col-self.filter_col)/self.stride+1)
		# final_row=int((self.in_row-self.filter_row)/self.stride+1)
		base_row=int(self.filter_row/2)
		base_col=int(self.filter_col/2)
		def fun(k,row_c,col_c,i):
			basey=row_c*self.stride
			topy=basey+self.filter_row
			basex=col_c*self.stride
			topx=basex+self.filter_col

			x=X[k,:,basey:topy,basex:topx]
			return np.sum(x*self.weights[i])+self.biases[i]
		###############################################
		# TASK 1 - YOUR CODE HERE

		temp=[[[[fun(k,m,n,i) for n in range(self.out_col)] for m in range(self.out_row)] for i in range(self.out_depth)] for k in range(n)]

		if self.activation == 'relu':
			# temp+=self.biases 
			return relu_of_X(temp)

			# raise NotImplementedError
		elif self.activation == 'softmax':
			return softmax_of_X(temp)

			# raise NotImplementedError

		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()

		
		###############################################

	def backwardpass(self, lr, activation_curr,activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size
		dep_prev=activation_prev.shape[1]
		row_prev=activation_prev.shape[1]
		col_prev=activation_prev.shape[1]
		dep_curr=activation_curr.shape[1]
		row_curr=activation_curr.shape[1]
		col_curr=activation_curr.shape[1]
		del_new=np.zeros((n,self.in_depth,self.in_row,self.in_col))
		del_w=np.zeros_like(self.weights)
		del_b=np.zeros_like(self.biases)

		###############################################
		# TASK 2 - YOUR CODE HERE
		if self.activation == 'relu':
			inp_delta=delta*gradient_relu_of_X(activation_curr,delta,1)		
			
			# temp=self.weights[None,:,:,:,:]
			# temp2=np.tile(temp,(n,1,1,1,1))
			# for h in range(self.out_row):
			# 	for w in range(self.out_col):
			# 		basey=h*self.stride
			# 		topy=basey+self.filter_row
			# 		basex=w*self.stride
			# 		topx=basex+self.filter_col
			# 		temp3=inp_delta[:,:,h,w]
			# 		temp3=temp3[:,:,None,None,None]
			# 		temp3=np.tile(temp3,(1,1,self.in_depth,self.filter_row,self.filter_col))

			# 		del_new[:,:,basey:topy,basex:topx]+=np.sum(temp2*temp3,axis=1)
			# 		temp4=activation_prev[:,None,:,basey:topy,basex:topx]
			# 		temp4=np.tile(temp4,(1,self.out_depth,1,1,1))
			# 		del_w+=lr*np.sum(temp3*temp4,axis=0)
			# self.weights-=del_w
			# inp_delta=gradient_relu_of_X(activation_curr,delta)*delta		
			# for i in range(self.out_depth):
				# self.biases[i]-=lr*np.sum(inp_delta[:,i,:,:])
			# for i in range(n):
			# temp1=self.weights
			for c in range(self.out_depth):
				for h in range(self.out_row):
					for w in range(self.out_col):
						basey=h*self.stride
						topy=basey+self.filter_row
						basex=w*self.stride
						topx=basex+self.filter_col
						temp=inp_delta[:,c,h,w]
						temp=temp[:,None,None,None]
						temp=np.tile(temp,(1,self.in_depth,self.filter_row,self.filter_col))
						del_w[c]+=np.sum(temp*activation_prev[:,:,basey:topy,basex:topx],axis=0)
						# del_w[c]+=lr*inp_delta[:,c,h,w]*
						temp2=self.weights[c]
						temp2=temp2[None,:,:,:]
						temp2=np.tile(temp2,(n,1,1,1))
						del_new[:,:,basey:topy,basex:topx]+=temp2*temp
			self.biases-=lr*np.sum(inp_delta,axis=(0,2,3))
			self.weights-=lr*del_w			
			return del_new
			raise NotImplementedError
		elif self.activation == 'softmax':
			inp_delta=gradient_softmax_of_X(activation_curr,delta)		
			for i in range(self.out_depth):
				self.biases[i]-=lr*np.sum(inp_delta[:,i,:,:])			
			for c in range(self.out_depth):
				for h in range(self.out_row):
					for w in range(self.out_col):
						basey=h*self.stride
						topy=basey+self.filter_row
						basex=w*self.stride
						topx=basex+self.filter_col
						temp=inp_delta[:,c,h,w]
						temp=temp[:,None,None,None]
						temp=np.tile(temp,(1,self.in_depth,self.filter_row,self.filter_col))
						del_w[c]+=lr*np.sum(temp*activation_prev[:,:,basey:topy,basex:topx],axis=0)
						# del_w[c]+=lr*inp_delta[:,c,h,w]*
						temp2=self.weights[c]
						temp2=temp2[None,:,:,:]
						temp2=np.tile(temp2,(n,1,1,1))
						del_new[:,:,basey:topy,basex:topx]+=temp2*temp
			self.weights-=del_w
			return del_new
			raise NotImplementedError
		else:
			print("ERROR: Incorrect activation specified: " + self.activation)
			exit()
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]
		# base_row=int(self.filter_row/2)
		# base_col=int(self.filter_col/2)
		def fun(k,row_c,col_c,i):
			basey=row_c*self.stride
			topy=basey+self.filter_row
			basex=col_c*self.stride
			topx=basex+self.filter_col

			x=X[k,i,basey:topy,basex:topx]
			return np.mean(x)
		###############################################
		# TASK 1 - YOUR CODE HERE

		temp=[[[[fun(k,m,n,i) for n in range(self.out_col)] for m in range(self.out_row)] for i in range(self.out_depth)] for k in range(n)]
		return np.array(temp)


		###############################################
		# TASK 1 - YOUR CODE HERE
		raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_curr,activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size
		del_new=np.zeros((n,self.in_depth,self.in_row,self.in_col))
		# del_w=np.zeros_like(self.weights)
		# del_b=np.zeros_like(self.biases)

		###############################################
		# TASK 2 - YOUR CODE HERE
		# if self.activation == 'relu':
		# inp_delta=delta*gradient_relu_of_X(activation_curr,delta,1)		
		# for i in range(n):
		# 	for c in range(self.out_depth):
		for h in range(self.out_row):
			for w in range(self.out_col):
				basey=h*self.stride
				topy=basey+self.filter_row
				basex=w*self.stride
				topx=basex+self.filter_col
				d =delta[:,:,h,w]/(self.filter_row*self.filter_col)

				del_new[:,:,basey:topy,basex:topx]+=d[:,:,None,None]
				# self.weights-=lr*delta[i][c][h][w]*activation_prev[i,:,basey:topy,basex:topx]
		# for i in range(self.out_depth):
			# self.biases[i]-=lr*np.sum(delta[:,i,:,:])
		return del_new

		###############################################
		# TASK 2 - YOUR CODE HERE
		raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, x,activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Function for the activation and its derivative
def relu_of_X(X):

	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu
	def fun(a):
		return max(0,a)
	return np.vectorize(fun)(X)
	
	raise NotImplementedError
	
def gradient_relu_of_X(X, delta,choose=0):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation relu amd during backwardpass
	if choose==0:
		def fun(a):
			if a>=0:
				return 1
			else:
				return 0
		return np.vectorize(fun)(X)
	else:
		def fun(a):
			if a>0:
				return 1
			else:
				return 0
		return np.vectorize(fun)(X)

	raise NotImplementedError
	
def softmax_of_X(X):
	# Input
	# data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
	# Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax
	temp=np.exp(X)
	temp1=temp@np.ones((X.shape[1],1))
	return temp/temp1
	raise NotImplementedError
	
def gradient_softmax_of_X(X, delta):
	# Input
	# data : Output from next layer/input | shape: batchSize x self.out_nodes
	# delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
	# Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
	# This will only be called for layers with activation softmax amd during backwardpass
	# Hint: You might need to compute Jacobian first
	def fun(i,j,k):
		if i==j:
			return X[k][i]*(1-X[k][j])
		else:
			return -X[k][i]*X[k][j]
	temp=np.array([(delta[k]@[[fun(i,j,k) for i in range(X.shape[1])] for j in range(X.shape[1])]).flatten() for k in range(X.shape[0])])
	return temp

	
	raise NotImplementedError
	