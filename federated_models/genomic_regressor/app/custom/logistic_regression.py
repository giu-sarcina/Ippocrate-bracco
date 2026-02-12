import torch
import torch.nn as nn

class LogisticRegressor(nn.Module):
    def __init__(self, input_dim, seed=0):
        super(LogisticRegressor, self).__init__()
        self.input_dim = input_dim
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # Initialize weights using Xavier/Glorot initialization
        self.weights = nn.Parameter(
            torch.randn((input_dim, 1), dtype=torch.float32),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        
    def forward(self, X):
        # X.shape should be (n, input_dim)
        return torch.matmul(X, self.weights) + self.bias

class BCEWithRegularization(nn.Module):
    def __init__(self, model, lambda_l2=0, pos_weight=None):
        super(BCEWithRegularization, self).__init__()
        self.model = model
        self.lambda_l2 = lambda_l2
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, predictions, targets):
        bce = self.bce_loss(predictions, targets)
        # Apply L2 regularization only to weights
        l2_reg = torch.sum(self.model.weights ** 2)
        return bce + self.lambda_l2 * l2_reg