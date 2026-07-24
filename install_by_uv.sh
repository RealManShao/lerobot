uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
sudo apt install ffmpeg
uv pip install -e ".[all]"
cd ..
cd tron2_env
uv pip install -e .