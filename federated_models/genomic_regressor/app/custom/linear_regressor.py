import torch
import torch.nn as nn

class LinearKernelRegressor(nn.Module):
    def __init__(self, input_dim, seed=0):
        super(LinearKernelRegressor, self).__init__()
        self.input_dim = input_dim
        torch.manual_seed(seed)
        self.weights = nn.Parameter(torch.randn((input_dim, 1), dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32), requires_grad=True)
        
    def forward(self, X):
        # X.shape should be (n, input_dim)
        linear_output = torch.matmul(X, self.weights) + self.bias
        return linear_output

class MSEWithRegularization(nn.Module):
    def __init__(self, model, lambda_l1=0, lambda_l2=0):
        super(MSEWithRegularization, self).__init__()
        self.model = model
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        # Apply regularization only to weights, not bias
        l1_reg = torch.sum(torch.abs(self.model.weights))
        l2_reg = torch.sum(self.model.weights ** 2)
        return mse + self.lambda_l1 * l1_reg + self.lambda_l2 * l2_reg

if __name__=="__main__":

    import pandas as pd

    print("Reading Genomic Data..")
    mydata = pd.read_csv('sample1_matrix.txt', sep='\t', engine='python')
    data = mydata.set_index('#Sample')
    print(data)
    print("Converting to Scalable..")
    array = data.to_numpy()

    # Hyperparameters
    input_size = len(array)
    output_size = 1
    lambda_l2 = 0.01
    
    # Select model
    print("Select Model..")
    model = LinearKernelRegressor(input_size)
    for p in model.parameters(): 
        print(p)
        
    # Reglarization
    criterion = MSEWithRegularization(model, lambda_l2=lambda_l2)
    
    # Mock Data
    print("Getting Label..")
    inputs = torch.tensor([array])
    targets = torch.randn(1, output_size)

    # Forward step
    print("Get prediction..")
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    
    print("Prediction:", predictions)
    print("LOSS:", loss)            
