import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
from PIL import Image


class BaseDataset(Dataset):
    """
    Base dataset class that handles augmentations.
    Child classes must implement _load_data(idx) method.
    """
    def __init__(self, augment=False, crop_size=None, crops_per_image=1):
        """
        Args:
            augment: Whether to apply data augmentations
            crop_size: Tuple (H, W) for random crops. If None, no cropping is applied.
            crops_per_image: Number of random crops to generate per image (only if crop_size is set)
        """
        self.augment = augment
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.files = []  # To be populated by child classes
        
    def __len__(self):
        # If we're generating multiple crops per image, multiply the length
        if self.crop_size is not None and self.crops_per_image > 1:
            return len(self.files) * self.crops_per_image
        return len(self.files)
    
    def __getitem__(self, idx):
        # Map index to file (account for multiple crops per image)
        if self.crop_size is not None and self.crops_per_image > 1:
            file_idx = idx // self.crops_per_image
        else:
            file_idx = idx
        
        # Load data from child class implementation
        image, target = self._load_data(file_idx)
        
        # Apply augmentations if enabled
        if self.augment:
            image, target = self._apply_augmentations(image, target)
        elif self.crop_size is not None:
            # Apply center crop for validation/test
            image, target = self._center_crop(image, target, self.crop_size)
            
        return image, target
    
    def _load_data(self, idx):
        """
        Load and return image and target as tensors.
        Must be implemented by child classes.
        
        Returns:
            image: torch.Tensor (C, H, W)
            target: torch.Tensor (C, H, W)
        """
        raise NotImplementedError("Child classes must implement _load_data()")
    
    def _apply_augmentations(self, image, target):
        """Apply random augmentations to image and target."""
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            target = TF.vflip(target)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle)
            target = TF.rotate(target, angle)
        
        # Random crop
        if self.crop_size is not None:
            i, j, h, w = self._get_random_crop_params(image, self.crop_size)
            image = TF.crop(image, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
        
        # Random brightness and contrast (only apply to image, not target)
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Add Gaussian noise to image only
        if random.random() > 0.5:
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(image) * noise_std
            image = image + noise
            image = torch.clamp(image, 0, 1)  # Ensure values stay in valid range
        
        return image, target
    
    def _get_random_crop_params(self, image, crop_size):
        """Get parameters for random crop."""
        h, w = image.shape[-2:]
        crop_h, crop_w = crop_size
        
        if h < crop_h or w < crop_w:
            raise ValueError(f"Crop size {crop_size} is larger than image size ({h}, {w})")
        
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)
        
        return i, j, crop_h, crop_w
    
    def _center_crop(self, image, target, crop_size):
        """Apply center crop to image and target."""
        h, w = image.shape[-2:]
        crop_h, crop_w = crop_size
        
        if h < crop_h or w < crop_w:
            raise ValueError(f"Crop size {crop_size} is larger than image size ({h}, {w})")
        
        i = (h - crop_h) // 2
        j = (w - crop_w) // 2
        
        image = TF.crop(image, i, j, crop_h, crop_w)
        target = TF.crop(target, i, j, crop_h, crop_w)
        
        return image, target


class NPZDataset(BaseDataset):
    """Dataset for loading .npz files."""
    
    def __init__(self, npz_dir, augment=False, crop_size=None, crops_per_image=1):
        super().__init__(augment, crop_size, crops_per_image)
        self.npz_dir = npz_dir
        self.files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    
    def _load_data(self, idx):
        """Load data from npz file."""
        file_path = os.path.join(self.npz_dir, self.files[idx])
        data = np.load(file_path)
        
        image = data['image'].astype(np.float32)
        target = data['target'].astype(np.float32)
        
        # Convert to tensors and ensure channel dimension
        if image.ndim == 2:
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        elif image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        else:
            image = torch.from_numpy(image)
            
        if target.ndim == 2:
            target = torch.from_numpy(target).unsqueeze(0)  # (1, H, W)
        elif target.ndim == 3 and target.shape[0] not in [1, 3, 6]:
            target = torch.from_numpy(target).permute(2, 0, 1)  # (C, H, W)
        else:
            target = torch.from_numpy(target)
        
        return image, target


class PNGDataset(BaseDataset):
    """Dataset for loading PNG files."""
    
    def __init__(self, images_dir, labels_dir, augment=False, crop_size=None, crops_per_image=1):
        super().__init__(augment, crop_size, crops_per_image)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        # Get sorted file/directory lists
        self.files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        self.label_dirs = sorted([d for d in os.listdir(labels_dir) 
                                 if os.path.isdir(os.path.join(labels_dir, d))])
    
    def _load_data(self, idx):
        """Load data from PNG files."""
        # Load image
        img_path = os.path.join(self.images_dir, self.files[idx])
        image = Image.open(img_path)
        image = np.array(image).astype('float32') / 255.0
        
        # Load labels (multiple PNGs per label directory)
        label_dir_path = os.path.join(self.labels_dir, self.label_dirs[idx])
        label_images = []
        for label_file in sorted(os.listdir(label_dir_path)):
            label_path = os.path.join(label_dir_path, label_file)
            label_image = Image.open(label_path)
            label_images.append(np.array(label_image).astype('float32') / 255.0)
        
        labels = np.stack(label_images)  # (C, H, W)
        
        # Convert to tensors
        if image.ndim == 2:
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        elif image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        else:
            image = torch.from_numpy(image)
        
        labels = torch.from_numpy(labels)  # Already (C, H, W)
        
        return image, labels


# def get_dataloaders(data_dir, dataset_type='npz', batch_size=8, val_split=0.1, test_split=0.1,
#                    crop_size=None, crops_per_image=1, seed=None, num_workers=0,
#                    labels_dir=None):
#     """
#     Unified function to create train, validation, and test dataloaders.
#     
#     Args:
#         data_dir: Directory containing data files (npz files or images for PNG)
#         dataset_type: Type of dataset - 'npz' or 'png'
#         batch_size: Batch size for dataloaders
#         val_split: Fraction of data for validation
#         test_split: Fraction of data for test
#         crop_size: Tuple (H, W) for random crops during training
#         crops_per_image: Number of crops to generate per image during training
#         seed: Random seed for reproducibility
#         num_workers: Number of worker processes for data loading
#         labels_dir: Directory containing labels (required for PNG dataset type)
#         
#     Returns:
#         train_loader, val_loader, test_loader
#     """
#     # Validate inputs
#     if dataset_type not in ['npz', 'png']:
#         raise ValueError(f"dataset_type must be 'npz' or 'png', got '{dataset_type}'")
#     
#     if dataset_type == 'png' and labels_dir is None:
#         raise ValueError("labels_dir is required when dataset_type='png'")
#     
#     # Create datasets based on type
#     if dataset_type == 'npz':
#         train_dataset_full = NPZDataset(
#             data_dir, 
#             augment=True, 
#             crop_size=crop_size, 
#             crops_per_image=crops_per_image
#         )
#         val_test_dataset = NPZDataset(
#             data_dir, 
#             augment=False, 
#             crop_size=crop_size, 
#             crops_per_image=1
#         )
#         num_files = len([f for f in os.listdir(data_dir) if f.endswith('.npz')])
#         
#     else:  # PNG
#         train_dataset_full = PNGDataset(
#             data_dir, 
#             labels_dir, 
#             augment=True, 
#             crop_size=crop_size, 
#             crops_per_image=crops_per_image
#         )
#         val_test_dataset = PNGDataset(
#             data_dir, 
#             labels_dir, 
#             augment=False, 
#             crop_size=crop_size, 
#             crops_per_image=1
#         )
#         num_files = len([f for f in os.listdir(data_dir) if f.endswith('.png')])
#     
#     # Calculate splits
#     test_size = int(test_split * num_files)
#     val_size = int(val_split * num_files)
#     train_size = num_files - test_size - val_size
#     
#     print(f"Files - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
#     if crops_per_image > 1:
#         print(f"Total training samples (with {crops_per_image} crops per image): {train_size * crops_per_image}")
#     
#     # Set random seed
#     if seed is not None:
#         torch.manual_seed(seed)
#         random.seed(seed)
#     
#     # Split file indices
#     file_indices = list(range(num_files))
#     random.shuffle(file_indices)
#     
#     train_file_indices = file_indices[:train_size]
#     val_file_indices = file_indices[train_size:train_size + val_size]
#     test_file_indices = file_indices[train_size + val_size:]
#     
#     # Create crop indices for training (multiply by crops_per_image)
#     if crops_per_image > 1:
#         train_indices = []
#         for file_idx in train_file_indices:
#             for crop_idx in range(crops_per_image):
#                 train_indices.append(file_idx * crops_per_image + crop_idx)
#     else:
#         train_indices = train_file_indices
#     
#     # Val/test indices remain the same (1 crop per image)
#     val_indices = val_file_indices
#     test_indices = test_file_indices
#     
#     # Create subset datasets
#     train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
#     val_dataset = torch.utils.data.Subset(val_test_dataset, val_indices)
#     test_dataset = torch.utils.data.Subset(val_test_dataset, test_indices)
#     
#     persistent_workers = num_workers > 0
#     
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         num_workers=num_workers, 
#         persistent_workers=persistent_workers
#     )
#     
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=num_workers, 
#         persistent_workers=persistent_workers
#     )
#     
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=batch_size, 
#         shuffle=False, 
#         num_workers=num_workers, 
#         persistent_workers=persistent_workers
#     )
#     
#     return train_loader, val_loader, test_loader


def get_dataloaders(data_dir, dataset_type='npz', batch_size=8, val_split=0.1, test_split=0.1,
                   crop_size=None, crops_per_image=1, seed=None, num_workers=0,
                   labels_dir=None):
    """
    Unified function to create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing data files (npz files or images for PNG)
        dataset_type: Type of dataset - 'npz' or 'png'
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        test_split: Fraction of data for test
        crop_size: Tuple (H, W) for random crops during training
        crops_per_image: Number of crops to generate per image during training
        seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        labels_dir: Directory containing labels (required for PNG dataset type)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Validate inputs
    if dataset_type not in ['npz', 'png']:
        raise ValueError(f"dataset_type must be 'npz' or 'png', got '{dataset_type}'")
    
    if dataset_type == 'png' and labels_dir is None:
        raise ValueError("labels_dir is required when dataset_type='png'")
    
    # Create datasets based on type
    if dataset_type == 'npz':
        train_dataset_full = NPZDataset(
            data_dir, 
            augment=True, 
            crop_size=crop_size, 
            crops_per_image=crops_per_image)

        num_files = len([f for f in os.listdir(data_dir) if f.endswith('.npz')]) * crops_per_image
        
    else:  # PNG
        train_dataset_full = PNGDataset(
            data_dir, 
            labels_dir, 
            augment=True, 
            crop_size=crop_size, 
            crops_per_image=crops_per_image
        )

        num_files = len([f for f in os.listdir(data_dir) if f.endswith('.png')]) * crops_per_image
    
    # Calculate splits
    test_size = int(test_split * num_files)
    val_size = int(val_split * num_files)
    train_size = num_files - test_size - val_size
    
    print(f"Files - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    
    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    # Split file indices
    file_indices = list(range(num_files))
    random.shuffle(file_indices)
    
    train_indices = file_indices[:train_size]
    val_indices = file_indices[train_size:train_size + val_size]
    test_indices = file_indices[train_size + val_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(train_dataset_full, val_indices)
    test_dataset = torch.utils.data.Subset(train_dataset_full, test_indices)
    
    persistent_workers = num_workers > 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader, test_loader