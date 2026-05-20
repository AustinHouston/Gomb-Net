import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, alpha=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  # Weight for penalizing false positives (alpha=1 is neutral, alpha>1 penalizes false positives, alpha<1 penalizes false negatives)

    def forward(self, input, target):
        input = input.view(-1)  # Flatten the input
        target = target.view(-1)  # Flatten the target
        intersection = torch.sum(input * target)
        # Apply a weight to the prediction sum to penalize false positives
        weighted_union = self.alpha * torch.sum(input) + torch.sum(target)
        dice_coeff = (2. * intersection + self.smooth) / (weighted_union + self.smooth)
        dice_coeff = torch.clamp(dice_coeff, 0, 1)
        return 1. - dice_coeff

class GombinatorialLoss(nn.Module):
    def __init__(self, group_size, loss='Dice', epsilon=1e-6, class_weights=None, alpha=2, sim_penalty=True):
        super(GombinatorialLoss, self).__init__()
        self.group_size = group_size
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.loss = loss.lower()
        self.alpha = alpha # for Dice loss
        self.sim_penalty = sim_penalty

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)
        total_loss = 0.0

        # Apply sigmoid to outputs if using Dice loss
        if self.loss == 'dice':
            outputs = torch.sigmoid(outputs)

        dice_loss = DiceLoss(alpha=self.alpha) if self.loss == 'dice' else None

        for i in range(batch_size):
            outputs_group1, outputs_group2 = outputs[i, :self.group_size], outputs[i, self.group_size:]
            targets_group1, targets_group2 = targets[i, :self.group_size], targets[i, self.group_size:]

            if self.loss == 'dice':
                loss00 = dice_loss(outputs_group1, targets_group1)
                loss01 = dice_loss(outputs_group1, targets_group2)
                loss10 = dice_loss(outputs_group2, targets_group1)
                loss11 = dice_loss(outputs_group2, targets_group2)
                output_loss = dice_loss(outputs_group1, outputs_group2) # maybe the penatly should be different
                target_loss = dice_loss(targets_group1, targets_group2) # for these two (modify alhpha)
            else:
                loss00 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss01 = F.cross_entropy(outputs_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                loss10 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group1, weight=self.class_weights, reduction='mean')
                loss11 = F.cross_entropy(outputs_group2.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')
                output_loss = F.cross_entropy(outputs_group1.unsqueeze(0), outputs_group2, weight=self.class_weights, reduction='mean')
                target_loss = F.cross_entropy(targets_group1.unsqueeze(0), targets_group2, weight=self.class_weights, reduction='mean')

            # Compute the inverse of loss pairings and sum for current sample
            inverse_loss = 1 / (loss01 + loss10 + self.epsilon) + 1 / (loss00 + loss11 + self.epsilon)
            prediction_loss = 1 / (inverse_loss + self.epsilon)

            if self.sim_penalty:
                # Loss penalizing similar predictions for G1 and G2
                mse_loss = (target_loss - output_loss) ** 2
                mse_loss = torch.sigmoid(mse_loss)

                # Accumulate the loss
                total_loss += prediction_loss * mse_loss
            else:
                # Accumulate the loss
                total_loss += prediction_loss

        # Average the accumulated losses over the batch
        return total_loss / batch_size


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss.
    Useful for better gradient flow, especially early in training.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            smooth: Smoothing factor for Dice loss
        """
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W) - logits if using BCEWithLogitsLoss
            target: Target output (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        # Apply sigmoid to predictions for Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        
        dice_loss = self.dice(pred_sigmoid, target)
        bce_loss = self.bce(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class MSEDiceLoss(nn.Module):
    """
    Combined Dice Loss and MSE Loss.
    Useful for regression tasks where you want both pixel-wise accuracy and overlap.
    """
    def __init__(self, dice_weight=0.5, mse_weight=0.5, smooth=1e-6):
        """
        Args:
            dice_weight: Weight for Dice loss component
            mse_weight: Weight for MSE loss component
            smooth: Smoothing factor for Dice loss
        """
        super(MSEDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.dice = DiceLoss(smooth=smooth)
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W)
            target: Target output (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        dice_loss = self.dice(pred, target)
        mse_loss = self.mse(pred, target)
        
        return self.dice_weight * dice_loss + self.mse_weight * mse_loss


class WeightedMSELoss(nn.Module):
    """
    MSE Loss with higher weight on non-background pixels.
    Good for Gaussian peaks on low background.
    """
    def __init__(self, background_weight=1.0, peak_weight=10.0, threshold=0.1):
        """
        Args:
            background_weight: Weight for background pixels
            peak_weight: Weight for peak/Gaussian pixels
            threshold: Threshold to distinguish background from peaks
        """
        super(WeightedMSELoss, self).__init__()
        self.background_weight = background_weight
        self.peak_weight = peak_weight
        self.threshold = threshold
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W)
            target: Target output (B, C, H, W)
        """
        # Create weight map: higher weight where target > threshold
        weight_map = torch.where(
            target > self.threshold,
            torch.tensor(self.peak_weight, device=target.device),
            torch.tensor(self.background_weight, device=target.device)
        )
        
        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = mse * weight_map
        
        return weighted_mse.mean()


class FocalMSELoss(nn.Module):
    """
    MSE Loss with focal weighting - automatically focuses on hard pixels.
    Good when you want the model to focus on poorly predicted regions.
    """
    def __init__(self, gamma=2.0):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W)
            target: Target output (B, C, H, W)
        """
        mse = (pred - target) ** 2
        
        # Focal weight: higher weight for larger errors
        focal_weight = mse ** (self.gamma / 2)
        
        weighted_mse = mse * focal_weight
        
        return weighted_mse.mean()


class L1Loss(nn.Module):
    """
    L1 (Mean Absolute Error) Loss.
    More robust to outliers than MSE, good for Gaussian prediction.
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.mae(pred, target)


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (Huber Loss).
    Combines benefits of L1 and L2: quadratic for small errors, linear for large.
    Excellent for regression with outliers.
    """
    def __init__(self, beta=1.0):
        """
        Args:
            beta: Threshold where loss transitions from quadratic to linear
        """
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, pred, target):
        return self.smooth_l1(pred, target)


class PeakAwareLoss(nn.Module):
    """
    Combined loss focusing on both overall accuracy and peak accuracy.
    Specifically designed for Gaussian peaks on background.
    """
    def __init__(self, mse_weight=0.3, peak_mse_weight=0.7, threshold=0.1):
        """
        Args:
            mse_weight: Weight for overall MSE
            peak_mse_weight: Weight for peak region MSE
            threshold: Threshold to identify peak regions
        """
        super(PeakAwareLoss, self).__init__()
        self.mse_weight = mse_weight
        self.peak_mse_weight = peak_mse_weight
        self.threshold = threshold
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W)
            target: Target output (B, C, H, W)
        """
        # Overall MSE
        overall_mse = F.mse_loss(pred, target)
        
        # MSE only on peak regions
        peak_mask = target > self.threshold
        if peak_mask.sum() > 0:
            peak_mse = F.mse_loss(pred[peak_mask], target[peak_mask])
        else:
            peak_mse = torch.tensor(0.0, device=pred.device)
        
        return self.mse_weight * overall_mse + self.peak_mse_weight * peak_mse


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss.
    Good for preserving structure and shape of Gaussians.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted output (B, C, H, W)
            target: Target output (B, C, H, W)
        """
        (_, channel, _, _) = pred.size()
        
        if channel == self.channel and self.window.dtype == pred.dtype:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(pred.device).type(pred.dtype)
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(pred, target, window, self.window_size, channel, self.size_average)
    
    def _ssim(self, pred, target, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred*pred, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target*target, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred*target, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class CombinedGaussianLoss(nn.Module):
    """
    Best overall loss for Gaussian prediction: combines MSE with peak awareness.
    """
    def __init__(self, mse_weight=0.5, peak_weight=0.5, threshold=0.1, peak_boost=5.0):
        """
        Args:
            mse_weight: Weight for overall MSE component
            peak_weight: Weight for peak-focused component
            threshold: Threshold to identify peak regions
            peak_boost: Multiplier for peak region importance
        """
        super(CombinedGaussianLoss, self).__init__()
        self.mse_weight = mse_weight
        self.peak_weight = peak_weight
        self.threshold = threshold
        self.peak_boost = peak_boost
    
    def forward(self, pred, target):
        # Overall MSE
        mse = F.mse_loss(pred, target)
        
        # Weighted MSE emphasizing peaks
        peak_mask = target > self.threshold
        weight_map = torch.ones_like(target)
        weight_map[peak_mask] = self.peak_boost
        
        weighted_mse = ((pred - target) ** 2 * weight_map).mean()
        
        return self.mse_weight * mse + self.peak_weight * weighted_mse
