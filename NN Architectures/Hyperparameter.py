import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import optuna
import argparse
import os

# Import your segmentation model.
from HybridEfficientNetUNetModel import EfficientNetUNet
from HybridResNetREDNetModel import ResNetREDNet
from HybridEfficientNetREDNetModel import EfficientNetREDNet
# Import your dataset.
from DiffusionDataset import DiffusionDataset




def train_one_epoch(model, train_loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    criterion = nn.L1Loss()
    model.train()
    running_loss = 0.0
    for diffused, clean in train_loader:
        diffused, clean = diffused.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(diffused)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate(model, val_loader, device):
    """
    Evaluates the model on the validation set.
    """
    criterion = nn.L1Loss()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for diffused, clean in val_loader:
            diffused, clean = diffused.to(device), clean.to(device)
            outputs = model(diffused)
            loss = criterion(outputs, clean)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def objective(trial):
    # Hyperparameter suggestions.
    
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    batch_size = trial.suggest_int("batch_size", 4, 16, step=4)
    num_epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetREDNet().to(device)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    # Data transforms.
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Use the dataset directories and cap value (set via command-line args).
    global diffused_dir, clean_dir, cap
    dataset = DiffusionDataset(diffused_dir, clean_dir, transform=transform, cap=cap)
    
    # Split dataset into training (80%) and validation (20%).
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop: run one epoch at a time and report validation loss.
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Trial {trial.number} Epoch {epoch+1}/{num_epochs} => Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Report the current validation loss to Optuna.
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

if __name__ == "__main__":
    global args

    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--grit_value", type=str, default="default_grit", help="Grit value for dataset directory")
    parser.add_argument("--cap", type=str, default="default_cap", help="Cap value for dataset")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of Parameters to optimize for")
    
    args = parser.parse_args()
    # Set up dataset paths.
    grit_value = args.grit_value
    cap = args.cap
    diffused_dir = os.path.join(".", "DMD", f"{grit_value} GRIT")
    clean_dir = os.path.join(".", "DMD", "Raw")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials)
    
    print("Best trial:")
    best_trial = study.best_trial
    print("  Validation Loss: {:.4f}".format(best_trial.value))
    print("  Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
