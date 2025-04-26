# from app.streamlit.main import streamlit_app

# if __name__ == '__main__':
#     streamlit_app()

import torch

# Check if GPU is available
print("GPU available:", torch.cuda.is_available())