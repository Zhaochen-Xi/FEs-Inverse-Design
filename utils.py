import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
import numpy as np
import os
import random
from tqdm import tqdm

def get_train_test_loader(dataset, batch_size=16, train_ratio=0.8, 
                          num_workers=1, pin_memory=True, random_seed=42):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

class LoadDataset(Dataset):
    """
    Generic Dataset Loader.
    Note: Modify _process_data method to fit your specific data format.
    """
    def __init__(self, root_dir, file_ext='.npz', random_seed=42, shuffle=True):
        self.root_dir = root_dir
        self.file_ext = file_ext
        self.random_seed = random_seed
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"{self.root_dir} does not exist.")

        self.file_list = [f for f in os.listdir(self.root_dir) if f.endswith(self.file_ext)]
        
        if not self.file_list:
            raise FileNotFoundError(f"No {self.file_ext} files found in {self.root_dir}.")

        if shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        
        try:
            raw_data = np.load(file_path, allow_pickle=True)
            data_tensor, label_tensor = self._process_data(raw_data)
            return data_tensor, label_tensor, file_name
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return dummy data or handle error appropriately
            return torch.zeros(1), torch.zeros(1), file_name

    def _process_data(self, raw_data):
        """
        Abstract method to process raw data.
        TO USE: Implement your specific extraction logic here.
        """
        # --- Example Implementation (Generic) ---
        # Assuming raw_data has keys 'x' and 'y'
        # data = raw_data['x']
        # labels = raw_data['y']
        
        # --- Placeholder for specific logic ---
        # Replace this with your specific key extraction 
        # e.g., polar_x = raw_data['domain'][:, 0]...
        
        raise NotImplementedError("You must implement _process_data in your local version or subclass.")
        
        # return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# Training loops remain mostly unchanged, just ensured imports are clean
def train_epoch(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader, desc="Training", leave=False):
        data_input = data[0].to(device, non_blocking=True)
        label_input = data[1].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(data_input, label_input)
        loss_dict = model.loss_function(*outputs, weight=0.05)
        loss = loss_dict['Loss']
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if scheduler:
        scheduler.step()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validation", leave=False):
            data_input = data[0].to(device, non_blocking=True)
            label_input = data[1].to(device, non_blocking=True)
            outputs = model(data_input, label_input)
            loss_dict = model.loss_function(*outputs, weight=0.05)
            loss = loss_dict['Loss']
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_cvae(model, train_dataloader, test_dataloader, epochs, learning_rate, device, save_dir='checkpoints', warmup_epochs=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    device = torch.device(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Tracking metrics
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = validate_epoch(model, test_dataloader, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save logic (simplified)
        if val_loss < best_val_loss and epoch >= warmup_epochs:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    return train_losses, val_losses