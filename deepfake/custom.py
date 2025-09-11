"""
PLEASE DO 
pip install transformers[torch] datasets evaluate torchvision
IF YOU HAVE NOT DONE PREVIOUSLY
Dataset:
https://huggingface.co/datasets/OpenRL/DeepFakeFace/tree/main

Unpack this and place the inpainting & wiki in the deepfake folder
"""
import os
import glob
import shutil
import random
import torch
import numpy as np
import evaluate 
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from torchvision.transforms import (
    ColorJitter,
    RandomHorizontalFlip,
    RandomResizedCrop
)

def reorganize_data_for_training(
    source_root, 
    output_data_path, 
    label_mapping, 
    validation_split_ratio=0.1
):
    """
    Description
    -----------
    Organizes images from source folders into a structured ImageFolder format.
    """

    print("--- Starting Data Reorganization ---")

    # check if already done and skip 
    if os.path.exists(os.path.join(output_data_path, "train", "fake")) and \
       os.path.exists(os.path.join(output_data_path, "train", "real")):
        print(f"Reorganized data folder '{output_data_path}' already exists. Skipping reorganization.")
        print("Delete the folder if you wish to re-process the data.")
        return

    # dir creation 
    train_path = os.path.join(output_data_path, "train")
    val_path = os.path.join(output_data_path, "validation")
    os.makedirs(os.path.join(train_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(train_path, "fake"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "real"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "fake"), exist_ok=True)

    total_images_processed = 0
    # actual organizing logic 
    for source_folder, label_name in label_mapping.items():
        print(f"\nProcessing source folder: '{source_folder}' -> label: '{label_name}'")
        folder_path = os.path.join(source_root, source_folder)

        if not os.path.isdir(folder_path):
            print(f"Warning: Source directory not found: {folder_path}. Skipping.")
            continue
        
        image_files = []
        for ext in ["*.jpg"]:
            image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

        if not image_files:
            print(f"Warning: No images found in {folder_path} and its subfolders.")
            continue

        # shuffle
        random.shuffle(image_files)
        split_index = int(len(image_files) * (1.0 - validation_split_ratio))
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        print(f"Found {len(image_files)} images. Splitting into {len(train_files)} train and {len(val_files)} validation.")

        for file_path in train_files:
            unique_filename = f"{source_folder}_{os.path.relpath(file_path, folder_path).replace(os.sep, '_')}"
            dest_path = os.path.join(train_path, label_name, unique_filename)
            shutil.copy(file_path, dest_path)

        for file_path in val_files:
            unique_filename = f"{source_folder}_{os.path.relpath(file_path, folder_path).replace(os.sep, '_')}"
            dest_path = os.path.join(val_path, label_name, unique_filename)
            shutil.copy(file_path, dest_path)

        total_images_processed += len(image_files)

    print(f"\nReorganization complete. Processed {total_images_processed} images.")
    print(f"Data ready at: {output_data_path}")

def train_model(
    data_folder_path, 
    model_name, 
    output_model_dir
):
    """
    Description
    -----------
    Loads the prepared data and trains the image classification model.
    """
    print("\n--- Starting Model Training ---")
    # data loader
    print(f"Loading dataset from local ImageFolder: {data_folder_path}")
    dataset = load_dataset("imagefolder", data_dir=data_folder_path)
    train_ds = dataset['train']
    val_ds = dataset['validation']
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    print(f"Loading processor for model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, cache_dir = "./cache")
    size = processor.size.get('shortest_edge', processor.size.get('height'))
    
    # data augmentation
    train_transforms = torch.nn.Sequential(
        RandomResizedCrop(size),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    )

    def apply_train_transforms(e):
        e['pixel_values'] = [
            processor(train_transforms(image.convert("RGB")), return_tensors="pt")['pixel_values'].squeeze()
            for image in e['image']
        ]
        return e

    def apply_val_transforms(e):
        e['pixel_values'] = [
            processor(image.convert("RGB"), return_tensors="pt")['pixel_values'].squeeze()
            for image in e['image']
        ]
        return e

    train_ds.set_transform(apply_train_transforms)
    val_ds.set_transform(apply_val_transforms)

    # load efficientnet-b1
    print(f"Loading pre-trained model: {model_name}")
    labels = train_ds.features['label'].names
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    training_args = TrainingArguments(
        output_dir=output_model_dir,
        
        # optimized for rtx 3050 laptop 6gb 
        per_device_train_batch_size=8,  
        gradient_accumulation_steps=4, 
        # mixed precision 
        fp16=True if torch.cuda.is_available() else False, 
        
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        logging_steps=20, 
    )

    def collate_fn(e):
        pixel_values = torch.stack([ex["pixel_values"] for ex in e])
        labels = torch.tensor([ex["label"] for ex in e])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    print("Starting model training process...")
    trainer.train()
    print(f"Training complete. Saving model to {output_model_dir}")
    trainer.save_model(output_model_dir)

if __name__ == "__main__":
    label_config = {
        "wiki": "real",      
        "inpainting": "fake"  
    }
    source_data_dir = "./deepfake" 
    reorganized_data_dir = "./deepfake_data_reorganized" 
    output_model_dir = "./efficientnet-b1-deepfake-detector"
    model = "google/efficientnet-b1"

    reorganize_data_for_training(source_data_dir, reorganized_data_dir, label_config)
    train_model(reorganized_data_dir, model, output_model_dir)