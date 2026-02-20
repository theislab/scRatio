from tqdm import tqdm 
import numpy as np 

def train(batch_size, n_steps, model, optimizer, X, C):
    """Standard training loop for scRatio 
    """
    for k in tqdm(range(n_steps)):
        optimizer.zero_grad()
    
        indices = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
        loss = model.shared_step(X[indices], C[indices], k)
        
        loss.backward()
        optimizer.step()
    return model
