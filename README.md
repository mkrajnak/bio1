# Neovascularization detection

### Dependencies
opencv-python
numpy

### Setup 
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage 
```
python3 neodetect.py <folder> [--debug]
<folder> - mandatory, folder with .jpg, RGB images of fundus
[--debug] - optional, debug images will be printed during image processing
```

