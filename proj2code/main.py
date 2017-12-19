
"""
Created on Mon Oct 23 2017
@author: dhanjal(50248990),d25(50248725)

"""


import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from decimal import *

# Computes the design matrix
# returns N X M design matrix
def compute_design_matrix(X, centers, spreads):
    basis_func_outputs = np.exp(np.sum(np.matmul(X - centers, spreads) * (X - centers), axis=2)/(-2)).T
    return np.insert(basis_func_outputs, 0, 1, axis=1)

# Computes closed form solution
# returns 1 X M closed form solution Wml
def closed_form_sol(L2_lambda, design_matrix, output_data): 
    return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix)
        ,np.matmul(design_matrix.T, output_data) ).flatten()

# Computes the validation error by creating design matrix for validation data and calculating Erms
# returns the predicted output for validation data
def validation_error(validation_input, validation_output, centers, spreads, Wml):
    basis_func = compute_design_matrix(validation_input, centers, spreads)
    predicted_output = np.matmul(basis_func, Wml.T)
    return get_rms_error(predicted_output, validation_output,Wml,0)

# Calculates the error while training
# returns the predicted output for training error
def train_error(design_matrix, train_output,Wml):
    predicted_output = np.matmul(design_matrix, Wml.T)
    return get_rms_error(predicted_output, train_output,Wml,1)

# Computes the stochastic gradient
# returns 1 X M sgd solution for Wml
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, weights):
    N, _ = design_matrix.shape
    #For plotting the behaviour of cost function with no of iterations with different values of learning rate
    # plt.figure();
    # plt.title('Letor Data - Graph of cost function with learning_rate 0.06');
    # plt.ion()
    # plt.ylabel('Cost Function')
    # plt.xlabel('# of Iterations')
    for epoch in range(num_epochs):
        for i in range(int(N/minibatch_size)):
            lower_bound = i * minibatch_size 
            upper_bound = min((i+1)*minibatch_size, N)
            Phi = design_matrix[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            error = train_error(Phi, output_data, weights)
            E_D = np.matmul((np.matmul(Phi, weights.T)-t).T, Phi)
            E = (E_D + L2_lambda * weights)/minibatch_size
            weights = weights - learning_rate * E 
            # print(np.linalg.norm(E))
            # cost_function_D = (t - np.matmul(Phi, weights.T)) ** 2
            # cost_function = 0.5 * (cost_function_D[:,0].sum())/N
            # plt.scatter(epoch, cost_function)
            # plt.pause(0.05)
    # plt.pause(0.05)
    # plt.ioff()
    # plt.show()
    return (error, weights)

# Divides the data set into train(80%), validation(10%) and test(10%)
def divide_dataset(input_set, output_set):
    lower_bound = 0;
    upper_bound = int(input_set.shape[0] * 0.80)

    train_input = input_set[lower_bound:upper_bound,:]
    train_output = output_set[lower_bound:upper_bound,:]

    lower_bound = upper_bound + 1
    upper_bound = int(input_set.shape[0] * 0.90)

    validation_input = input_set[lower_bound:upper_bound, :]
    validation_output = output_set[lower_bound:upper_bound,:]

    lower_bound = upper_bound + 1
    upper_bound = input_set.shape[0]

    test_input = input_set[lower_bound:upper_bound, :]
    test_output = output_set[lower_bound:upper_bound, :]

    return (train_input, train_output, validation_input, validation_output, test_input, test_output)

# returns the variance of dimensions D X D where D is the no of features for a given input set
def get_spreads(input_set):
    var = np.zeros((input_set.shape[1], input_set.shape[1]))
    for i in range(input_set.shape[1]):
        var[i,i] = (np.var(input_set[:,i]) + 0.001)
    return var

# Calculates centers using kmeans method and spreads for those clusters
# returns centers of dimensions M X 1 X D
# returns spreads of dimensions M X D X D
# M denotes the number of basis functions
# D denotes the no of features in a data point
def get_centers_and_spread(input_set, M):
    kmeans = cluster.KMeans(n_clusters = M).fit(input_set)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    centers = centers[:,np.newaxis,:]
    spreads = np.zeros((M, input_set.shape[1], input_set.shape[1]))
    
    # get variance for the points in respective clusters and stack in M x D x D matrix
    for i in range(M):
        inputs = input_set[np.where(labels==i)]
        sigma = get_spreads(inputs)
        spreads[i,:,:] = np.linalg.inv(sigma)
    return (centers, spreads)

# Adds a new axis to input values
def format_input(input_set):
    return input_set[np.newaxis,:,:]

# Calculates the rms error
# returns the Root mean square error of given input set and output set on tuned hyperparameters
def get_rms_error(predicted_output, output_set,wml,isTrain):
    NoOfTestingInp = output_set.shape[0]
    E_D = np.zeros([NoOfTestingInp,1])
    predicted_output = predicted_output[:,np.newaxis]
    E_D = np.square(output_set - predicted_output)
    # E = 0.5 * E_D.sum()

    # E_D = (output_set - predicted_output) ** 2
    if isTrain:
        E = 0.5 * (E_D[:,0].sum() + np.matmul(wml,wml.T));
    else:
        E = 0.5 * (E_D[:,0].sum())

    
    return (2*E/NoOfTestingInp)**0.5

# Train using early stop in stochastic gradient,num_epochs steps are calculated through this function
# returns the optimal steps calculated by tuning hyperparameters where validation error is minimum
# returns the weights after training on validation set
def check_early_stop(design_matrix, output_data, validation_input_stack, validation_output, centers, spreads):
    
    M = centers.shape[0]
    # number of times to check if the validation error is worsening
    patience_num = 10
    total_steps = 0
    Wml = 0
    optimal_steps = 0
    count = 0
    
    # randomly initiate weights
    weights = np.random.rand(1,M+1)
    v_min = float('inf')
    
    num_epochs = 10
    learning_rate = 0.001
    L2_lambda = 0.1
    i=0
    minibatch_size = design_matrix.shape[0]

    # return the optimum values when count exceeds the patience number
    while count < patience_num:
        # keeps track of number of epochs
        total_steps = total_steps + num_epochs

        # get the weights by running num_epoch times and get the validation error
        _, weights = SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, weights)
        v_error = validation_error(validation_input_stack, validation_output, centers, spreads, weights)
        
        # save weights and optimal steps when minimum validation error is obtained
        if v_error < v_min:
            count = 0
            Wml = weights
            optimal_steps = total_steps
            v_min = v_error
        else:
            count = count+1
    return optimal_steps, Wml.flatten()

# Calculate the no of basis functions M for which the training set error is minimum.
# Then calculate the corresponsing centers and spreads for the value of M obtained for further processing. 
# Calculates the closed form solution for a computed M.
# returns centers of dimension M X 1 X D
# returns spreads of dimension M X D X D
# returns weights of dimension 1 X M
def train_closed_form(input_set, output_set, L2_lambda,performTraining,noOfbiasFunctions):
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(input_set, output_set)
    
    M = 2
    centers_min = np.zeros(1)
    error_arr1 = np.zeros(24)
    spreads_min = np.zeros(1)
    weights = np.zeros(1)
    v_min = float('inf')
    design_matrix = np.zeros(1)
    
    train_input_stack = format_input(train_input)
    validation_input_stack = format_input(validation_input)
    i=0
       
    # grid search for M
    # start with M = 4 and increment by 2 in each iteration 
    if performTraining : 
        while M < 50:
            M = M + 2;
            # get centers using k-means
            centers, spreads = get_centers_and_spread(train_input, M)
            design_matrix = compute_design_matrix(train_input_stack, centers, spreads)

            # compute the weights and calculate validation error
            Wml = closed_form_sol(L2_lambda, design_matrix, train_output)
            cur_validation_error = validation_error(validation_input_stack, validation_output, centers, spreads, Wml)

            # save centers and spreads after finding the minimum valdiation error
            if cur_validation_error < v_min:
                centers_min = centers
                v_min = cur_validation_error
                spreads_min = spreads
            error_arr1[i] = cur_validation_error
            i = i+1
        
        
    else:
        M = noOfbiasFunctions;
        centers, spreads = get_centers_and_spread(train_input, M)
        centers_min = centers
        spreads_min = spreads

    design_matrix = compute_design_matrix(train_input_stack, centers_min, spreads_min)
    Wml = closed_form_sol(L2_lambda, design_matrix, train_output)
    return (centers_min, spreads_min, Wml,error_arr1)

# Test the closed form solution by giving test input and outputs as parameters
# returns predicted output on testing data for closed-form-solution of dimensions N X 1
def test_closed_form(test_input, test_output, centers_min, Wml, spreads):
    test_input_stack = format_input(test_input)
    test_design_matrix = compute_design_matrix(test_input_stack, centers_min, spreads)
    predicted_output = np.matmul(test_design_matrix, Wml.T)
    return predicted_output

# Feed the whole dataset_input, dataset_output, centers and spreads to train using stochastic gradient descent
# returns the weights optimized using training input and validation input data sets with dimensions 1 X M
def train_SGD(input_set, output_set, centers, spreads,isCheckEarlyStop,noOfbiasFunctions):
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(
        input_set, output_set)
    
    train_input_stack = format_input(train_input)
    validation_input_stack = format_input(validation_input)
    number = 110;
    design_matrix = compute_design_matrix(train_input_stack, centers, spreads)
    
    if isCheckEarlyStop:
        #calculate num_epochs value from early stop method.
        number, weights = check_early_stop(
        design_matrix, train_output, validation_input_stack, validation_output, centers, spreads)
    else:
        M = noOfbiasFunctions
        
        _, weights = SGD_sol(learning_rate=0.03, minibatch_size=design_matrix.shape[0], num_epochs=10, L2_lambda=0.1, design_matrix=design_matrix, output_data=train_output, weights= np.random.rand(1, M+1))
        v_error = validation_error(validation_input_stack, validation_output, centers, spreads, weights)
    

    #un-comment this and the plots lines in SGD _sol to see the output generation
    #Train hyperParameter Eta - learning rate by selecting different values in range 0.01 to 0.1
    # SGD_sol(learning_rate=0.01,minibatch_size=train_input_stack.shape[0],num_epochs=100,L2_lambda=0.1,design_matrix=design_matrix,output_data=train_output,weights = np.random.rand(1, M+1))
    print("Optimal Steps calculated for num_epochs",number)

    return (number, weights)

# Input the test inputs, test outputs, model parameters to predict the output
# returns predicted output on testing data of SGD with dimensions N X 1
def test_SGD(test_input, test_output, centers, Wml, spreads):
    test_input_stack = format_input(test_input)
    test_design_matrix = compute_design_matrix(test_input_stack, centers, spreads)
    predicted_output = np.matmul(test_design_matrix, Wml.T)
    return predicted_output


# Computes the closed form solution and Stochastic Gradient Descent of Synthetic Data
def process_synthetic_data(input_data,output_data):
	### synthetic data
	# call train_SGD() to train using stochastic gradient descent method
	# call test_SGD() to test for new input values
    L2_lambda = 0.1

    #divide the data into training set(80%),validation set(10%),testing set(10%)
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(input_data, output_data)
    print("Value of tuned hyperparameter Lambda is :",L2_lambda)
    print("Value of tuned hyperparameter bias function M is : 42")
    print("Value of tuned hyperparameter learning Parameter is :0.03");
    #calculate the centers,spreads,weights on training set and feed them to testing set after validation.
    #pass performTraining = 1 for whole computation and performTraining = 0 for fixed noofBiasFunctions passed as next parameter
    centers, spreads, Wml, error_arr = train_closed_form(input_data, output_data, L2_lambda, 0 , 34)

    predicted_output = test_closed_form(train_input, train_output, centers, Wml, spreads)
    Erms_train = get_rms_error(predicted_output, train_output,Wml,1)
    print("Training Erms of Closed Form Solution on Synthetic Data : ",Erms_train)

    predicted_output = test_closed_form(validation_input, validation_output, centers, Wml, spreads)
    Erms_val = get_rms_error(predicted_output, validation_output,Wml,0)
    print("Validation Erms of Closed Form Solution on Synthetic Data : ",Erms_val)

    #The predicted output of testing set on closed form solution
    predicted_output = test_closed_form(test_input, test_output, centers, Wml, spreads)
    Erms_test = get_rms_error(predicted_output, test_output,Wml,0)
    print("Testing Erms of Closed Form Solution on Synthetic Data : ",Erms_test)


    # # #plot the graph of predicted output and given output
    fig = plt.figure();
    plt.title('Predicted Output vs Given Output - Synthetic Data(closed Form)')
    T = range(0,predicted_output.shape[0])
    plt.plot(T,test_output,'b-');
    plt.plot(T,predicted_output,'r-');
    plt.xlabel("# Of Iterations");
    plt.ylabel("Predicted Output")
    plt.xticks(np.arange(0, predicted_output.shape[0], 500))
    plt.show()
    plt.close('all');
    plt.clf()
    #train the SGD data on M calculated from closed form solution.
    #set isCheckEarlystop = 1 inorder to calculate the number of epoches through early stop method else set is CheckEarlystop =0 
    #and the noOFBiasFunctions. We chose noOFBiasFunctions = 34 as estimated by our training
    centers, spreads, Wml, error_arr = train_closed_form(input_data, output_data, L2_lambda, 0 , 34)

    number, weights = train_SGD(input_data, output_data, centers, spreads,0,34)
    
    predicted_output = test_SGD(train_input, train_output, centers, weights, spreads)
    Erms_train_sgd = get_rms_error(predicted_output, train_output,weights,1)
    print("Training Erms of SGD on Synthetic Data",Erms_train_sgd)

    predicted_output = test_SGD(validation_input, validation_output, centers, weights, spreads)
    Erms_val_sgd = get_rms_error(predicted_output, validation_output,weights,0)
    print("Validation Erms of SGD on Synthetic Data",Erms_val_sgd)

    #predict the output of testing set as per the tuned hyper parameters
    predicted_output = test_SGD(test_input, test_output, centers, weights, spreads)
    Erms_test_sgd = get_rms_error(predicted_output, test_output,weights,0)
    print("Testing Erms of SGD on Synthetic Data",Erms_test_sgd)

    #uncomment this in order to plot the graphs of Predicted Output vs Given Output of test data

    # #plot the graph of predicted output and given output
    fig = plt.figure();
    plt.title('Predicted Output vs Given Output on Synthetic Data(SGD)')
    T = range(0,predicted_output.shape[0])
    plt.plot(T,test_output,'b-');
    plt.plot(T,predicted_output,'r-');
    plt.xlabel("# Of Iterations");
    plt.ylabel("Predicted Output")
    plt.xticks(np.arange(0, predicted_output.shape[0], 500))
    plt.show()
    plt.close('all');
    plt.clf()

    

# Computes the closed form solution and Stochastic Gradient Descent of LeTOR Data
def process_Letor_data(letor_input,letor_output):
	# LeTOR data
	# call train_closed_form() to train the linear regression model
    # call test_closed_form() to test for new input values
    L2_lambda = 0.1
    
    #divide the data into training set(80%),validation set(10%),testing set(10%)
    train_input, train_output, validation_input, validation_output, test_input, test_output = divide_dataset(letor_input, letor_output)
    print("Value of tuned hyperparameter Lambda is :",L2_lambda)
    print("Value of tuned hyperparameter bias function M is : 34")
    print("Value of tuned hyperparameter learning Parameter is :0.03");
    #calculate the centers,spreads,weights on training set and feed them to testing set after validation.
    #pass performTraining = 1 for whole computation and performTraining = 0 for fixed noofBiasFunctions passed as next parameter
    centers, spreads, Wml, error_arr = train_closed_form(letor_input, letor_output, L2_lambda,0,42)
    
    predicted_output = test_closed_form(train_input, train_output, centers, Wml, spreads)
    Erms_train = get_rms_error(predicted_output, train_output,Wml,1)
    print("Training Erms of Closed Form Solution on Letor Data : ",Erms_train)

    predicted_output = test_closed_form(validation_input, validation_output, centers, Wml, spreads)
    Erms_val = get_rms_error(predicted_output, validation_output,Wml,0)
    print("Validation Erms of Closed Form Solution on Letor Data : ",Erms_val)

    # #The predicted output of testing set on closed form solution
    

    predicted_output = test_closed_form(test_input, test_output, centers, Wml, spreads)
    Erms_test = get_rms_error(predicted_output, test_output,Wml,0)
    print("Testing Erms of Closed Form Solution on Letor Data: ",Erms_test)
    
    # #plot the graph of predicted output and given output
    fig = plt.figure();
    plt.title('Predicted Output vs Given Output - Letor Data(Closed Form)')
    T = range(0,predicted_output.shape[0])
    plt.plot(T,test_output,'b-');
    plt.plot(T,predicted_output,'r-');
    plt.xlabel("# Of Iterations");
    plt.ylabel("Predicted Output")
    plt.xticks(np.arange(0, predicted_output.shape[0], 500))
    plt.show()
    plt.close('all')
    plt.clf()
    #train the SGD data on M calculated from closed form solution.
    # set isCheckEarlystop = 1 inorder to calculate the number of epoches through early stop method else set is CheckEarlystop =0 
    # and the noOFBiasFunctions. We chose noOFBiasFunctions = 42 as estimated by our training
    centers, spreads, Wml, error_arr = train_closed_form(letor_input, letor_output, L2_lambda,0,42)
    number, weights = train_SGD(letor_input, letor_output, centers, spreads,0,42)

    predicted_output = test_SGD(train_input, train_output, centers, weights, spreads)
    Erms_train_sgd = get_rms_error(predicted_output, train_output,weights,1)
    print("Training Erms of SGD on Letor Data",Erms_train_sgd)

    predicted_output = test_SGD(validation_input, validation_output, centers, weights, spreads)
    Erms_val_sgd = get_rms_error(predicted_output, validation_output,weights,1)
    print("Validation Erms of SGD on Letor Data",Erms_val_sgd)

    #predict the output of testing set as per the tuned hyper parameters
    predicted_output = test_SGD(test_input, test_output, centers, weights, spreads)
    Erms_test_sgd = get_rms_error(predicted_output, test_output,weights,1)
    print("Testing Erms of SGD on Letor Data:",Erms_test_sgd)

    #uncomment this in order to plot the graphs of Predicted Output vs Given Output of test data
    #plot the graph of predicted output and given output
    fig = plt.figure();
    plt.title('Predicted Output vs Given Output on Synthetic Data(SGD)')
    T = range(0,predicted_output.shape[0])
    plt.plot(T,test_output,'b-');
    plt.plot(T,predicted_output,'r-');
    plt.xlabel("# Of Iterations");
    plt.ylabel("Predicted Output")
    plt.xticks(np.arange(0, predicted_output.shape[0], 500))
    plt.show()
    plt.close('all')
    plt.clf()

# Main function to start the computation of closed form and Stochastic Gradient Descent on different dataSets
def main():
	# load the input and output data
	input_data = np.loadtxt('datafiles/input.csv', delimiter=',')
	output_data = np.loadtxt('datafiles/output.csv', delimiter=',').reshape([-1,1])
	letor_input = np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
	letor_output = np.genfromtxt('datafiles/Querylevelnorm_t.csv').reshape([-1,1])
	#start the processing on different inputs
	process_synthetic_data(input_data,output_data)
	process_Letor_data(letor_input,letor_output)

# Trigger
main()

    
    

    
    

    
    
   
    