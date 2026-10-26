# AI Face Generator — DCGAN

A small, ready-to-run PyTorch project packaged around the supplied `generator.pth` checkpoint and example image. It uses a DCGAN-style generator to create 64×64 RGB images from random latent vectors.

## Project files

- `model.py` — generator neural-network architecture.
- `generator.pth` — supplied generator weights.
- `generate.py` — command-line image generation.
- `app.py` — Streamlit web interface with image downloads.
- `sample_input_image.png` — image supplied alongside the checkpoint, included as a reference.
- `requirements.txt` — Python dependencies.
- `outputs/` — generated images are saved here when you run the scripts.

## Requirements

- Python 3.10 or 3.11 recommended.
- Internet access the first time dependencies are installed.
- CPU works; a CUDA-compatible GPU can make generation faster.

## Setup (Windows PowerShell)

```powershell
cd face_generator_project
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`.

## Generate images from the command line

```powershell
python generate.py
```

Generate 5 images with a repeatable random seed:

```powershell
python generate.py --count 5 --seed 42 --output outputs
```

## Run the web app

```powershell
streamlit run app.py
```

Open the local address printed in the terminal by Streamlit. Choose the number of images and click **Generate images**. Use each download button to save a result.

## Notes

- The network structure is inferred from the tensor shapes in the supplied checkpoint and uses the standard 64×64 DCGAN generator layout. The included scripts load the checkpoint strictly and will report an error if its architecture differs.
- This project contains the generator only. It does not include the discriminator, training dataset, or training code, so it is for inference/generation rather than retraining.
- The model's output quality depends on the training data and training process. The supplied example image is included as a reference, not as a guarantee of output quality.
- Keep API keys and other credentials out of source code and ZIP files. Kaggle credential JSON files were intentionally not included.
