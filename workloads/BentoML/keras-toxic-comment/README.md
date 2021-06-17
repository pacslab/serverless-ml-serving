
Create conda environment:

```
conda create -n keras-toxic-comment python=3.6
conda activate keras-toxic-comment
conda install -c conda-forge gcc
pip install -r requirements.in
```


To compile the dependencies:


```
pip install pip-tools
pip-compile
```
