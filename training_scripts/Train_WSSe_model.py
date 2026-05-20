import torch
import sys
sys.path.append('/Users/austin/Documents/GitHub/Gomb-Net/')
from GombNet.utils import *
from GombNet.networks import *
from GombNet.loss_func import GombinatorialLoss

# Create dataloaders
images_dir='/Users/austin/Desktop/WSSe_dataset_8kb/images'
labels_dir='/Users/austin/Desktop/WSSe_dataset_8kb/labels'
train_loader, val_loader, test_loader = get_dataloaders(images_dir, dataset_type="png", labels_dir=labels_dir, batch_size = 8, val_split=0.2, test_split=0.1, num_workers=0)

# model_names = ['a', 'b', 'c', 'd', 'e', 'f']
model_names = ['b', 'c', 'd', 'e', 'f']
save_name_base = '/Users/austin/Desktop/WSSe_ensemble/WSSe_model_'

# Model Params
input_channels = 1
num_classes = 6    # number of output classes
num_filters = [32, 64, 128, 256, 512]

# Check if a GPU is available and if not, use a CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps') # backend for Apple silicon GPUs
    print('Using Apple silicon GPU')
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using device: {device}')

# Function to initialize the weights of the model
def initialize_model_weights(model, initialization_type):
    if initialization_type == 'xavier':
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    elif initialization_type == 'kaiming':
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
    elif initialization_type == 'orthogonal':
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    else:
        raise ValueError(f"Unknown initialization type: {initialization_type}")

    model.apply(init_weights)

# Loop through models and apply different initializations
# initialization_methods = ['xavier', 'kaiming', 'orthogonal', 'xavier', 'kaiming', 'orthogonal']
initialization_methods = ['kaiming', 'orthogonal', 'xavier', 'kaiming', 'orthogonal']


for model_name, init_method in zip(model_names, initialization_methods):
    # Create and train model with CombinatorialGroupLoss
    print(f'Loading model {model_name}...')
    model = TwoLeggedUnet(input_channels, num_classes, num_filters, dropout = 0.423)
    initialize_model_weights(model, init_method)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # 0.001
    loss = GombinatorialLoss(group_size = num_classes//2, loss = 'Dice', epsilon=1e-6, class_weights = None, alpha=2) # 1.58

    # Define the unique save name for this model
    save_name = save_name_base + model_name

    print(f'Training model {model_name}...')                     
    model, train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs = 100,
                                            criterion = loss, optimizer = optimizer, device = device,
                                            save_name = save_name, save_checkpoints=[10,30,50,100])
    
    print(f'Model {model_name} training complete and saved at {save_name}')

print("All models trained and saved.")