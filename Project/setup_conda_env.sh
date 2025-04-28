conda create --name HW4ML python=3.9 # type y so process would continue
conda activate HW4ML
pip install notebook
pip install tensorflow==2.15.0
python -c "import tensorflow as tf; assert tf.__version__ == '2.15.0', f'\n\n\nWrong TensorFlow version: {tf.__version__}'; print('\n\n\nTensorFlow version check passed\!')"
srun --time=01:00:00 --gres gpu:1 --mem=10G --resv-ports=1 --pty /bin/bash -l
hostname -I
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888