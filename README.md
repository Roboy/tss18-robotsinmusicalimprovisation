# Robots in musical improvisation

## Prerequisites 
1. Create virtual environment in the root folder of this project:
```bash
virtualenv .rimi -p python3 --no-site-packages
```
2. Activate virtual environment by:
```bash
source .rimi/bin/activate
```
3. Set PYTHONPATH to the root directory of this project or add to ~/.bashrc
```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/dir
```
4. Pip install all packages
```bash
pip3 install -r requirements.txt
```
5. Install PyTorch
If you don't need GPU Support or you are working on Linux with CUDA 9.0, try:
```bash
pip3 install torch torchvision
```
For other systems or CUDA versions, please visit https://pytorch.org/

6. TODO