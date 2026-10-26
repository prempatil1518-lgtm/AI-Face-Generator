# AI Face Generator Using Deep Learning

An AI-powered face image generation application built using PyTorch and a pretrained Generative Adversarial Network (GAN) generator. The application generates synthetic images from random noise and provides a simple web interface for viewing and downloading the generated images.

## Features

- Generate synthetic images using a pretrained generator model.
- Generate multiple images in a single run.
- Interactive web interface built with Streamlit.
- Download generated images directly from the application.
- Save generated images locally.
- Support for CPU and CUDA-enabled GPUs.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- Pillow
- Deep Learning
- Generative Adversarial Networks (GAN)

## Project Structure

```text
AI-Face-Generator/
│
├── app.py                  # Streamlit web application
├── model.py                # Generator model architecture
├── generate.py             # Image generation script
├── generator.pth           # Pretrained generator weights
├── sample_input_image.png  # Reference image
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── .gitignore              # Files excluded from Git
└── outputs/                # Generated images
