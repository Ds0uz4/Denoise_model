# **Denoise and Classify Project**

This project implements a deep learning pipeline to first denoise a set of noisy flower photos and then classify the denoised images into one of five flower categories. The approach was to build a robust, modular system, suitable for deployment and further development.

### **Table of Contents**

1. [Denoising Method](https://www.google.com/search?q=%23denoising-method)  
2. [Classification Approach](https://www.google.com/search?q=%23classification-approach)  
3. [Key Design Decisions](https://www.google.com/search?q=%23key-design-decisions)  
4. [Challenges and How I Overcame Them](https://www.google.com/search?q=%23challenges-and-how-i-overcame-them)  
5. [Repository Structure](https://www.google.com/search?q=%23repository-structure)  
6. [Requirements](https://www.google.com/search?q=%23requirements)  

### **Denoising Method**

The core of the denoising strategy is a **Denoising Autoencoder (DAE)**. It is inspired by the **U-Net** architecture and is designed to take a noisy image as input and reconstruct a clean version.

* **Encoder Path:** This path progressively downsamples the noisy image, learning to extract abstract and high-level features.  
* **Decoder Path:** This path reconstructs the image using upsampling layers and **skip connections** from the encoder. This is vital for retaining fine-grained spatial information, such as edges and textures, which results in a more accurate reconstruction.  
* **Residual Blocks:** To enhance the training of this deep network, I incorporated Residual Blocks, which help mitigate the vanishing gradient problem. The entire DAE was trained using the **Mean Squared Error (MSE)** loss function.

### **Classification Approach**

My approach to the classification task is a prime example of **cascaded learning**. This allows each model to specialize in a specific task, leading to better overall performance. I created a two-step pipeline:

* **Step 1: Denoising:** I use the trained DAE to denoise the noisy images from the test set. This preprocessing step cleans the data and removes noise artifacts that could be distracting for the classifier.  
* **Step 2: Classification:** The denoised images are then passed to a separate, dedicated classification model named FlowerClassifier. This is a deep convolutional neural network (CNN) with a feature extraction front-end and a fully-connected classifier head. I used **dropout layers** to prevent overfitting, and the model was trained using **Cross-Entropy Loss**.

### **Key Design Decisions**

* **Modular Code Structure:** I refactored the original Jupyter Notebook into a well-organized Python project with separate files for models, data loading, and utilities, creating a clean, readable, and reusable codebase.  
* **Cascaded Learning:** Separating the denoising and classification tasks into two distinct models allows each to specialize, leading to better overall performance.  
* **Hybrid Evaluation Metrics:** I went beyond just the training loss (MSE) to evaluate the DAE's performance, using **Peak Signal-to-Noise Ratio (PSNR)** and the **Structural Similarity Index (SSIM)** to provide a more complete view of image quality.  
* **Model Checkpointing:** I implemented model checkpointing to save pre-trained weights, making the training process more efficient and saving significant time during development.

### **Challenges and How I Overcame Them**

* **Normalization Mismatch:** There was a massive problem with the normalization of the dataset, where the DAE model worked without normalization, but the classifier model required it. This problem led to more bluish images in the denoised dataset. I resolved this by adding an option for normalization directly into the load\_data function, allowing for flexible data preparation tailored to each model's needs.

### **Repository Structure**

Denoise\_Project/  
├── main.py  
├── models.py  
├── data\_loader.py  
├── utils.py  
├── requirements.txt  
├── Denoised\_Images/

### **Requirements**

To run this project, you will need the following Python libraries. You can install them using pip:

kaggle  
scikit-image  
opencv-python  
pandas  
numpy  
matplotlib  
seaborn  
torch  
tqdm  
torchvision
