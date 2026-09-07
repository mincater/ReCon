"""Root entrypoint for Streamlit Community Cloud deployment."""

import os
import sys

# Ensure root directory is on Python path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.streamlit_app import main

if __name__ == "__main__":
    main()
