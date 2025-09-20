import os
import glob
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm.auto import tqdm

def reorganize_data_for_training(
    source_root,
    output_data_path,
    label_mapping,
    validation_split_ratio=0.1
):
    """
    Organizes images from source folders into a structured 'train' and 'validation'
    directory format, which is compatible with torchvision's ImageFolder.
    """
    print("--- Starting Data Reorganization ---")

    if os.path.exists(os.path.join(output_data_path, "train")) and \
       os.path.exists(os.path.join(output_data_path, "validation")):
        print(f"Reorganized data folder '{output_data_path}' already exists. Skipping reorganization.")
        print("Delete the folder if you wish to re-process the data.")
        return

    train_path = os.path.join(output_data_path, "train")
    val_path = os.path.join(output_data_path, "validation")
    for label_name in set(label_mapping.values()):
        os.makedirs(os.path.join(train_path, label_name), exist_ok=True)
        os.makedirs(os.path.join(val_path, label_name), exist_ok=True)

    total_images_processed = 0
    for source_folder, label_name in label_mapping.items():
        print(f"\nProcessing source folder: '{source_folder}' -> label: '{label_name}'")
        folder_path = os.path.join(source_root, source_folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Source directory not found: {folder_path}. Skipping.")
            continue

        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
        if not image_files:
            print(f"Warning: No images found in {folder_path}.")
            continue
            
        random.shuffle(image_files)
        split_index = int(len(image_files) * (1.0 - validation_split_ratio))
        train_files, val_files = image_files[:split_index], image_files[split_index:]
        print(f"Found {len(image_files)} images. Splitting into {len(train_files)} train and {len(val_files)} validation.")

        for file_path in train_files:
            unique_filename = f"{source_folder}_{os.path.relpath(file_path, folder_path).replace(os.sep, '_')}"
            shutil.copy(file_path, os.path.join(train_path, label_name, unique_filename))
        for file_path in val_files:
            unique_filename = f"{source_folder}_{os.path.relpath(file_path, folder_path).replace(os.sep, '_')}"
            shutil.copy(file_path, os.path.join(val_path, label_name, unique_filename))

        total_images_processed += len(image_files)

    print(f"\nReorganization complete. Processed {total_images_processed} images.")
    print(f"Data ready for training at: {output_data_path}")

def evaluate_and_visualize(model, val_loader, device, output_dir, class_names):
    """
    Generates a precision-recall curve and Grad-CAM visualizations after training.
    """
    print("\n--- Starting Post-Training Evaluation and Visualization ---")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # pr curve
    print("Generating Precision-Recall curve...")
    fake_class_index = class_names.index('fake')
    precision, recall, _ = precision_recall_curve(all_labels, all_probs[:, fake_class_index], pos_label=fake_class_index)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    print(f"Precision-Recall curve saved to {pr_curve_path}")

    # ---------------------------------------- gradcam visuzliation
    # gradcam
    print("\nGenerating Grad-CAM heatmaps for 10 random validation images...")
    grad_cam_output_dir = os.path.join(output_dir, "grad_cam_examples")
    os.makedirs(grad_cam_output_dir, exist_ok=True)
    
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)

    val_dataset = val_loader.dataset
    num_images = min(10, len(val_dataset))
    random_indices = random.sample(range(len(val_dataset)), num_images)

    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i, idx in enumerate(random_indices):
        input_tensor, true_label_int = val_dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        pred_class = model(input_tensor).argmax(axis=1).item()
        targets = [ClassifierOutputTarget(pred_class)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        rgb_img = inv_normalize(input_tensor.squeeze(0)).cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        true_label_text = class_names[true_label_int]
        pred_label_text = class_names[pred_class]
        
        save_path = os.path.join(grad_cam_output_dir, f"{i+1}_true-{true_label_text}_pred-{pred_label_text}.png")
        cv2.imwrite(save_path, vis_bgr * 255)
        
    print(f"Saved {num_images} Grad-CAM images to {grad_cam_output_dir}")

def train_model(data_folder_path, model_name, output_model_dir, num_epochs=1):
    """
    Loads data, creates a timm model, and runs a standard PyTorch training loop.
    """
    print("\n--- Starting Model Training with PyTorch and Timm ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_folder_path, x), data_transforms[x])
                      for x in ['train', 'validation']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'validation']}
    
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # --------------------------------- train loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloaders[phase], desc=phase.capitalize())
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("\nTraining complete.")
    os.makedirs(output_model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_model_dir, 'model_weights.pth'))
    print(f"Model saved to {output_model_dir}")

    evaluate_and_visualize(model, dataloaders['validation'], device, output_model_dir, class_names)

if __name__ == "__main__":
    source_data_dir = "/home/mms/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset/Train"
    label_config = {"Real": "real", "Fake": "fake"}
    reorganized_data_dir = "./deepfake_kaggle_reorganized"
    output_model_dir = "./efficientnet-b1-timm-deepfake-detector"
    model_name = "efficientnet_b1"

    if not os.path.isdir(source_data_dir):
        print("="*80)
        print(f"ERROR: Source data directory not found at: '{source_data_dir}'")
        print("Please update the 'source_data_dir' variable.")
        print("="*80)
    else:
        reorganize_data_for_training(source_data_dir, reorganized_data_dir, label_config)
        train_model(reorganized_data_dir, model_name, output_model_dir)

