import sys
sys.path.insert(0,'../')

from GombNet.utils import *
from GombNet.networks import *
from GombNet.loss_func import GombinatorialLoss, DiceLoss

# Create dataloaders
images_dir='/Users/austin/Desktop/gomb_beta/Graphene_dataset/images'
labels_dir='/Users/austin/Desktop/gomb_beta/Graphene_dataset/labels'
train_loader, val_loader, test_loader = get_dataloaders(images_dir, dataset_type="png", labels_dir=labels_dir, batch_size = 2, val_split=0.2, test_split=0.1)

# debug
# debug_loader = DataLoader(train_loader.dataset, batch_size=1, num_workers=0, shuffle=False)
# for i, sample in enumerate(debug_loader):
#     try:
#         # Print to see what comes out
#         print(f"Sample {i}: {sample}")
#     except Exception as e:
#         print(f"Error at sample {i}: {e}")
#         break


# Model Params
input_channels = 1
num_classes = 2    # number of output classes
num_filters = [32, 64, 128, 256] # for the 256 net
# num_filters = [64, 128, 256, 512] # for the 512 net

# Check if a GPU is available and if not, use a CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps') # backend for Apple silicon GPUs
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
# Create and train model with CombinatorialGroupLoss
print('Loading model...')
model = TwoLeggedUnet(input_channels, num_classes, num_filters, dropout = 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss = DiceLoss(alpha=1)
print('Training model...')                         
model, train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs = 100,
                                          criterion = loss, optimizer = optimizer, device = device,
                                          save_name = '/Users/austin/Desktop/gomb_beta/Dice_2Legs_G/model',
                                          save_checkpoints = [10,30,50])
