# Robots in Musical Improvisation

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
export PYTHONPATH=$PYTHONPATH:/path/to/dir
```
4. OPTIONAL (for Ubuntu): You will need these for python-rtmidi:
```bash
sudo apt-get install libasound-dev
sudo apt-get install libjack-dev
```
5. Pip install all packages
```bash
pip3 install -r requirements.txt
```
6. Install PyTorch 

If you are working on Ubuntu with CUDA 9.0, try:
```bash
pip3 install torch torchvision
```
For other systems or CUDA versions, please visit https://pytorch.org/

6. TODO