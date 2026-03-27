import torch.nn as nn
import torch.nn.functional as F
import torch 

class ProbabilisticClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        
        super(ProbabilisticClassifier, self).__init__()

        # Classifier layers 
        layers = []

        if num_hidden_layers == 0:
            # Single linear layer: input -> output
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First hidden layer
            layers += [nn.Linear(input_dim, hidden_dim), nn.ELU()]
            
            # Additional hidden layers
            for _ in range(num_hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
                
            # Output layer (no activation here — softmax applied in forward)
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
    
    def compute_log_ratio(self, x, y_idx, y_prime_idx):
        # Get the output probabilities
        probs = self.forward(x)
        probs = F.log_softmax(probs, dim=1)  # Apply softmax to get probabilities
        
        # Extract the probabilities for the specified classes
        p_y = probs[:, y_idx]
        p_y_prime = probs[:, y_prime_idx]
        
        # Compute the log ratio
        log_ratio = p_y - p_y_prime
        
        return log_ratio
