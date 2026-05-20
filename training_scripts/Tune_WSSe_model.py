import optuna
import torch
from GombNet.utils import *
from GombNet.networks import *
from GombNet.loss_func import GombinatorialLoss

# Create dataloaders
images_dir='/Users/austin/Desktop/WSSe_dataset/images'
labels_dir='/Users/austin/Desktop/WSSe_dataset/labels'
train_loader, val_loader, test_loader = get_dataloaders(images_dir, dataset_type="png", labels_dir=labels_dir, batch_size = 2, val_split=0.2, test_split=0.1)

save_name = '/Users/austin/Documents/GitHub/GombNet/trained_models/WSSe_dataset_32_1024.pth'

# Model Params
input_channels = 1
num_classes = 6

# Check if a GPU is available and if not, use a CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps') # backend for Apple silicon GPUs
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


print('tuning hyperparameters...')
# Objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    num_filters = trial.suggest_categorical('num_filters', [[64, 128, 256, 512, 1024], [32, 64, 128, 256, 512], [32, 64, 128, 256, 512, 1024]])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 1, 5)

    # Model initialization
    model = TwoLeggedUnet(input_channels, num_classes, num_filters, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = GombinatorialLoss(group_size=num_classes // 2, loss='Dice', epsilon=1e-6, class_weights=None, alpha=alpha)

    # Training with pruning
    n_epochs = 100
    for epoch in range(n_epochs):
        model, train_loss, val_loss = train_model(
            model,
            train_loader,
            val_loader,
            n_epochs=1,  # Train for one epoch at a time
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            save_name=None,
            save_loss_history=False
        )

        val_loss = val_loss[-1]
        # Report intermediate value
        trial.report(val_loss, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

# Run Optuna optimization with ASHA pruner
study = optuna.create_study(
    direction='minimize', 
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
)
study.optimize(objective, n_trials=50)

# Best hyperparameters
best_params = study.best_params
print('Best hyperparameters:', best_params)

# Train final model with best hyperparameters
best_num_filters = best_params['num_filters']
best_dropout = best_params['dropout']
best_lr = best_params['lr']
best_alpha = best_params['alpha']

model = TwoLeggedUnet(input_channels, num_classes, best_num_filters, dropout=best_dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
criterion = GombinatorialLoss(group_size=num_classes // 2, loss='Dice', epsilon=1e-6, class_weights=None, alpha=best_alpha)


print('Training model with best hyperparameters...')
model, train_loss, val_loss = train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=100,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    save_name=save_name
)

# Save the final model
torch.save(model.state_dict(), save_name)
print(f'Model saved to {save_name}')