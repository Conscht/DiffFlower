import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import fiftyone as fo

class ClassicLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassicLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomTorchImageDataset(Dataset):
    def __init__(
        self,
        fiftyone_dataset,
        base_transforms=None,
        idk_transforms=None,
        label_map=None,
        gt_field="ground_truth",
        idk_label_str="IDK",
    ):
        """
        Args:
            fiftyone_dataset: The FiftyOne dataset/view to load from
            base_transforms: Transforms to apply to all images
            idk_transforms: Additional transforms for IDK class only
            label_map: Dict mapping string labels to integer indices
            gt_field: Field name containing the labels (e.g. 'curated_label')
            idk_label_str: String label that represents the IDK class
        """
        # Note: We do NOT store self.fiftyone_dataset because it contains MongoDB connections
        # which cannot be pickled for multiprocessing (num_workers > 0) on Windows.
        # Instead, we immediately extract the list of filepaths and labels.
        self.image_paths = fiftyone_dataset.values("filepath")
        self.str_labels = fiftyone_dataset.values(f"{gt_field}.label")
        
        self.base_transforms = base_transforms
        self.idk_transforms = idk_transforms
        self.idk_label_str = idk_label_str
        self.label_map = label_map

        print(
            f"CustomTorchImageDataset initialized with {len(self.image_paths)} samples "
            f"(gt_field='{gt_field}', idk_label='{self.idk_label_str}')."
        )
        
        # Validate that we have labels for all images
        if not self.str_labels or len(self.str_labels) != len(self.image_paths):
            print("WARNING: Number of labels doesn't match number of images!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("L")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return dummy data to avoid crashing
            return torch.zeros(1, 28, 28), torch.tensor(0, dtype=torch.long)
            
        label_str = self.str_labels[idx]

        # Apply base transforms to all samples
        if self.base_transforms is not None:
            image = self.base_transforms(image)

        # Apply additional transforms only for IDK samples
        if (self.idk_transforms is not None) and (label_str == self.idk_label_str):
            image = self.idk_transforms(image)

        # Get label index
        if self.label_map:
            label_idx = self.label_map.get(label_str, -1)
            if label_idx == -1:
                label_idx = 0 
        else:
            try:
                label_idx = int(label_str)
            except:
                label_idx = 0
                
        return image, torch.tensor(label_idx, dtype=torch.long)

def train_epoch(model, train_loader, optimizer, criterion, device):
    batch_losses = []
    model.train()

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss_value = criterion(logits, labels)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        batch_losses.append(loss_value.item())

    return np.mean(batch_losses)


def val_epoch(model, val_loader, criterion, device, return_accuracy=False):
    batch_losses = []
    correct = 0
    total = 0

    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss_value = criterion(logits, labels)
            batch_losses.append(loss_value.item())

            if return_accuracy:
                _, preds = torch.max(logits, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

    val_loss = np.mean(batch_losses)

    if return_accuracy:
        return val_loss, correct / total

    return val_loss

def evaluate_idk_performance(model, val_loader, device, label_map):
    """
    Runs full evaluation on the validation loader and prints a detailed
    performance table including IDK vs Digit confusion metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("Running inference on validation set...")
    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix
    # Rows: Ground Truth, Cols: Predicted
    n_classes = len(label_map)
    idk_index = label_map.get("IDK", -1)
    if idk_index == -1:
        print("Error: 'IDK' not found in label_map. Cannot print IDK stats.")
        return

    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(all_labels, all_preds):
        if t < n_classes and p < n_classes:
            confusion[t, p] += 1

    # ---------------------------------------------------------
    # Print Summary Table
    # ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"{'Class':<10} | {'Total GT':<10} | {'Correct':<8} | {'Pred as IDK':<12} | {'Accuracy':<8} | {'IDK Rate':<8}")
    print(f"{'-'*80}")

    total_correct = 0
    total_samples = 0

    # Sort by index
    sorted_map = sorted(label_map.items(), key=lambda x: x[1])

    for label_str, class_idx in sorted_map:
        n_samples = np.sum(confusion[class_idx, :])     # Total GT
        n_correct = confusion[class_idx, class_idx]     # Correct
        n_pred_idk = confusion[class_idx, idk_index]    # How many called IDK
        
        accuracy = (n_correct / n_samples * 100) if n_samples > 0 else 0.0
        idk_rate = (n_pred_idk / n_samples * 100) if n_samples > 0 else 0.0
        
        prefix = "> " if class_idx == idk_index else "  "
        
        row_str = (f"{prefix}{label_str:<8} | {n_samples:<10} | {n_correct:<8} | "
                   f"{n_pred_idk:<12} | {accuracy:6.2f}% | {idk_rate:6.2f}%")
        print(row_str)
        
        total_correct += n_correct
        total_samples += n_samples

    print(f"{'-'*80}")
    overall_acc = total_correct / total_samples * 100
    total_pred_idk = np.sum(confusion[:, idk_index])

    print(f"{'OVERALL':<10} | {total_samples:<10} | {total_correct:<8} | "
          f"{total_pred_idk:<12} | {overall_acc:6.2f}% | {'--':<8}")
    print(f"{'='*80}")

    # ---------------------------------------------------------
    # IDK Performance Metrics
    # ---------------------------------------------------------
    idk_tp = confusion[idk_index, idk_index]
    idk_fp = np.sum(confusion[:, idk_index]) - idk_tp
    idk_fn = np.sum(confusion[idk_index, :]) - idk_tp

    precision = idk_tp / (idk_tp + idk_fp) if (idk_tp + idk_fp) > 0 else 0
    recall = idk_tp / (idk_tp + idk_fn) if (idk_tp + idk_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nIDK Class Deep Dive:")
    print(f"  1. Precision: {precision:.2%} (When model says 'IDK', it is correct {precision:.2%} of the time)")
    print(f"  2. Recall:    {recall:.2%} (Model found {recall:.2%} of all real IDK samples)")
    print(f"  3. F1 Score:  {f1:.4f}")
