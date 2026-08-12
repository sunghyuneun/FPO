# Installation
If you know what you're doing (presumably if you're looking at this), do what you want. Personally, I used conda to manage Python version, even though I only used pip to install packages.

```bash
conda init
conda create --name FPO python=3.12
conda activate FPO

# From the root path, run
pip install -r requirements.txt
```