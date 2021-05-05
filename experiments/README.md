# Experimenter

To run a jupyter notebook while recording the new outputs:

```sh
jupyter nbconvert --to notebook --inplace --execute experiment.ipynb
# or the following if facing the kernel not found error:
# jupyter nbconvert --to notebook --inplace --ExecutePreprocessor.kernel_name=python3 --execute experiment.ipynb
# list of kernels:
#   jupyter kernelspec list
```
