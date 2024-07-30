The codebase is mainly built with the following libraries:

- Python 3.6 or higher

- TensorFlow:<br>
We can successfully reproduce the main results under two settings below:<br>
Tesla A100 (40G): CUDA 11.1 + TensorFlow 2.4.0<br>
Tesla V100 (32G): CUDA 10.1 + TensorFlow 2.3.0

- DeepSpeed
```python

DS_BUILD_OPS=1 pip install deepspeed`
```
- TensorBoard

- Decord

- Einops

### Installation command for tenorflow
```python
pip install tensorflow tensorflow-addons tensorflow-datasets tensorflow-hub tensorboard decord einops
DS_BUILD_OPS=1 pip install deepspeed  # Only if using DeepSpeed
```