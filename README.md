# Facial Age Prediction with Deep Learning

This project uses deep learning to predict a person’s age based on facial images. Leveraging transfer learning with ResNet50, the model is trained on a labeled dataset of facial photographs.

## 📊 Objective
To build a regression model that estimates a person's age from an image using a convolutional neural network (CNN).

## 🛠️ Tools & Technologies
- Python
- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- pandas, NumPy, OpenCV
- Matplotlib & Seaborn for visualization

## 📁 Dataset
- ChaLearn LAP dataset (`images/`, `labels.csv`)
- Contains ~7,600 labeled images with age annotations

## 🔍 Project Workflow
1. **Data Preprocessing**  
   - Loaded images and normalized pixel values  
   - Resized images to 224x224 for ResNet compatibility  
   - Merged labels and images

2. **Model Building**  
   - Used pre-trained ResNet50 as base  
   - Added custom fully-connected layers for regression  
   - Compiled with MSE loss and Adam optimizer

3. **Training**  
   - Trained on GPU using data augmentation and early stopping  
   - Evaluated using MAE (Mean Absolute Error)

4. **Results & Evaluation**  
   - Visualized training/validation loss  
   - Compared predicted vs actual age on test set

## 📈 Results
- Model achieved strong performance with low MAE  
- ResNet50 transfer learning significantly improved training efficiency and accuracy

## ▶️ How to Run
1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run `main.ipynb` to execute the full pipeline

## 🚧 Future Improvements
- Add gender as an additional input feature  
- Experiment with EfficientNet or deeper models  
- Improve data augmentation and error calibration

## ✅ Status
Project Completed — Model successfully predicts age from facial images.
