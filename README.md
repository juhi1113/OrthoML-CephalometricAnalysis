
# Cephalometric Analysis Using Deep Learning

## Introduction
This project aims to address the health concerns associated with traditional cephalometric radiographs in orthodontics by developing a non-invasive, radiation-free alternative. By leveraging deep learning techniques, we can analyze facial photographs to provide accurate cephalometric measurements, significantly reducing patient exposure to harmful radiation.

## Features
- **Deep Learning Model**: Utilizes transfer learning techniques to achieve an 82% accuracy rate in predicting cephalometric measurements.
- **User-Friendly Web Application**: A seamless interface allows users to upload facial photos and receive instant cephalometric measurements.
- **No Radiation Exposure**: A safer alternative to traditional x-ray methods, ensuring patient safety.
- **Scalable Solution**: Designed for potential integration into clinical practices.

## Demo


https://github.com/user-attachments/assets/62c0f15e-dcc4-449d-9544-3a4d0ddd731e



## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)

## Installation
1. **Clone the repository**:
    ```bash
    git clone (https://github.com/juhi1113/OrthoML-CephalometricAnalysis.git)
    cd cephalometric-analysis
    ```
2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Flask app**:
    ```bash
    python app.py
    ```
2. **Access the web application**:
    Open your web browser and go to `http://127.0.0.1:5000`.
3. **Upload a facial photograph**: 
    Use the web interface to upload an image and receive cephalometric measurements.

## Model Details
The model is based on a Convolutional Neural Network (CNN) architecture, utilizing transfer learning with a pre-trained ResNet18 model. It processes facial images to predict key cephalometric measurements. 
- **Dataset**: 400 facial images with annotated landmarks.
- **Model Accuracy**: Achieved an 82% accuracy rate.
- **Error Reduction**: Reduced landmark detection error by 30% through model optimization.

## Results
- **Accuracy**: 82% in predicting cephalometric measurements.
- **Error Reduction**: 30% reduction in landmark detection errors.
- **Impact**: Potential to replace traditional x-ray methods, enhancing patient safety.

## Technologies
- **Programming Languages**: Python
- **Frameworks and Libraries**: PyTorch, Flask, Jupyter Notebook, Pandas, NumPy
- **Web Development**: HTML, CSS, JavaScript

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
