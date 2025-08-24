import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import cv2

from data_loader import load_data, CLASSES
from models import DAE, FlowerClassifier
from utils import psnr_ssim_evaluation, normalize, denormalize

DATASET_PATH = "./REC_DATASET"
TRAIN_NOISY_DIR = os.path.join(DATASET_PATH, 'train/noisy')
TRAIN_CLEAN_DIR = os.path.join(DATASET_PATH, 'train/clean')
TEST_NOISY_DIR = os.path.join(DATASET_PATH, 'test/noisy')

DENOISED_OUTPUT_DIR = "Denoised_Images"
LABELS_OUTPUT_FILE = "test_labels.csv"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 75
CLASSIFIER_EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_kaggle_config():
    
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    if os.path.exists("kaggle.json"):
        shutil.copy("kaggle.json", kaggle_dir)
        try:
            os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
        except:
            pass
    print("Kaggle setup complete.")

def download_and_extract_data():
    print("Assuming dataset is already downloaded and extracted.")

def train_dae_model(model, dataloader, loss_fn, optimizer):
    """Trains the Denoising Autoencoder model."""
    print("--- Training Denoising Autoencoder ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for noisy_images, clean_images in tqdm(dataloader, desc=f"DAE Epoch {epoch+1}/{EPOCHS}"):
            noisy_images = noisy_images.to(DEVICE)
            clean_images = clean_images.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = loss_fn(outputs, clean_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "dae_model.pth")
    print("DAE model saved.")

def train_classifier_model(classifier_model, classifier_loader, criterion, classifier_optimizer):
    """Trains the Flower Classifier model."""
    print("--- Training Flower Classifier ---")
    for epoch in range(CLASSIFIER_EPOCHS):
        classifier_model.train()
        Loss = 0
        for images, labels in tqdm(classifier_loader, desc=f"Classifier Epoch {epoch+1}/{CLASSIFIER_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            classifier_optimizer.zero_grad()
            outputs = classifier_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            classifier_optimizer.step()
            Loss += loss.item()
        epoch_loss = Loss / len(classifier_loader)
        print(f"Epoch :{epoch+1}, loss : {epoch_loss:.4f}")

    torch.save(classifier_model.state_dict(), "classifier_model.pth")
    print("Classifier model saved.")

def denoise_and_classify(dae_model, classifier_model, test_images, test_filenames):
    """Denoises test images and classifies them."""
    dae_model.eval()
    classifier_model.eval()

    denoised_test_images_list = []
    predicted_classes_list = []

    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(test_images), batch_size=BATCH_SIZE, shuffle=False)

        for (batch,) in tqdm(test_loader, desc="Denoising and Classifying Images"):
            batch = batch.to(DEVICE)
            denoised_batch = dae_model(batch)
     
            denoised_for_classifier = torch.stack([normalize(img) for img in denoised_batch])
            
            outputs = classifier_model(denoised_for_classifier)
            _, predicted = torch.max(outputs, 1)

            denoised_test_images_list.append(denoised_batch.cpu())
            predicted_classes_list.append(predicted.cpu())

    denoised_images_tensor = torch.cat(denoised_test_images_list, dim=0)
    predicted_classes = torch.cat(predicted_classes_list, dim=0).numpy() + 1
    

    if not os.path.exists(DENOISED_OUTPUT_DIR):
        os.makedirs(DENOISED_OUTPUT_DIR)

    for i, denoised_image in enumerate(denoised_images_tensor.numpy()):
        denoised_image = np.transpose(denoised_image, (1, 2, 0))
        denoised_image = (denoised_image * 255).astype(np.uint8)
        output_path = os.path.join(DENOISED_OUTPUT_DIR, test_filenames[i])
        cv2.imwrite(output_path, cv2.cvtColor(denoised_image, cv2.COLOR_RGB2BGR))
    
    print(f"Denoised images saved to {DENOISED_OUTPUT_DIR}")

    results_df = pd.DataFrame({'Image_File_Name': test_filenames, 'Predicted_Label': predicted_classes})
    results_df.to_csv(LABELS_OUTPUT_FILE, index=False)
    print(f"Predictions saved to {LABELS_OUTPUT_FILE}")

def main():
    clean_images, clean_images_labels, _ = load_data(TRAIN_CLEAN_DIR, is_labeled=True, is_normalize=False)
    clean_images_normalized, _, _ = load_data(TRAIN_CLEAN_DIR, is_labeled=True, is_normalize=True)
    noisy_images, _, _ = load_data(TRAIN_NOISY_DIR, is_labeled=True, is_normalize=False)
    test_images, _, test_filenames = load_data(TEST_NOISY_DIR, is_labeled=False, is_normalize=False)

    clean_images_tensor = torch.tensor(clean_images, dtype=torch.float32)
    clean_images_normalized_tensor = torch.tensor(clean_images_normalized, dtype=torch.float32)
    clean_images_labels_tensor = torch.tensor(clean_images_labels, dtype=torch.long)
    noisy_images_tensor = torch.tensor(noisy_images, dtype=torch.float32)
    test_images_tensor = torch.tensor(test_images, dtype=torch.float32)

    denoising_model = DAE().to(DEVICE)
    dae_loss_fn = nn.MSELoss()
    dae_optimizer = torch.optim.Adam(denoising_model.parameters(), lr=0.001)

    dae_dataset = TensorDataset(noisy_images_tensor, clean_images_tensor)
    dae_dataloader = DataLoader(dae_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if os.path.exists("dae_model.pth"):
        print("Loading DAE model from checkpoint...")
        denoising_model.load_state_dict(torch.load("dae_model.pth"))
    else:
        train_dae_model(denoising_model, dae_dataloader, dae_loss_fn, dae_optimizer)
    

    classifier_model = FlowerClassifier(num_classes=len(CLASSES)).to(DEVICE)
    classifier_criterion = nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)

    classifier_dataset = TensorDataset(clean_images_normalized_tensor, clean_images_labels_tensor)
    classifier_loader = DataLoader(classifier_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if os.path.exists("classifier_model.pth"):
        print("Loading Classifier model from checkpoint...")
        classifier_model.load_state_dict(torch.load("classifier_model.pth"))
    else:
        train_classifier_model(classifier_model, classifier_loader, classifier_criterion, classifier_optimizer)
        
    # --- Evaluation ---
    denoise_and_classify(denoising_model, classifier_model, test_images_tensor, test_filenames)


    avg_psnr, avg_ssim = psnr_ssim_evaluation(noisy_images, clean_images)
    print(f"Original Noisy vs Clean Images - Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()