"""Simple Streamlit interface for the supplied DCGAN generator."""
from pathlib import Path
import streamlit as st
import torch
from torchvision.utils import save_image
from model import Generator

st.set_page_config(page_title="AI Face Generator", page_icon="🧑", layout="centered")
st.title("AI Face Generator")
st.write("Generate 64×64 synthetic images using the included pretrained DCGAN generator checkpoint.")
st.caption("This is a generative model demo. Outputs may look distorted and are not guaranteed to be realistic.")

@st.cache_resource
def load_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator().to(device)
    checkpoint = Path(__file__).with_name("generator.pth")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device

count = st.slider("Number of images", min_value=1, max_value=8, value=1)
seed = st.number_input("Random seed (use the same seed to reproduce results)", min_value=0, max_value=2147483647, value=42)
if st.button("Generate images", type="primary"):
    model, device = load_generator()
    torch.manual_seed(int(seed))
    with torch.no_grad():
        noise = torch.randn(count, 100, 1, 1, device=device)
        generated = model(noise).cpu()
    cols = st.columns(min(count, 4))
    for i, img in enumerate(generated):
        path = Path("outputs")
        path.mkdir(exist_ok=True)
        out_file = path / f"streamlit_generated_{i+1:03d}.png"
        save_image(img, out_file, normalize=True)
        with cols[i % len(cols)]:
            st.image(str(out_file), caption=f"Generated image {i+1}", use_container_width=True)
            st.download_button(
                f"Download image {i+1}",
                data=out_file.read_bytes(),
                file_name=out_file.name,
                mime="image/png",
                key=f"download_{i}_{seed}",
            )
