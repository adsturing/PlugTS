## Code Structure

* `main.py`: the main function for the experiment
* `PlugTS_P.py`: the PlugTS_P algorithm
* `PlugTS.py`: the PlugTS algorithm
* `data.py`: the data preprocessing script

## Dataset

* For choosing a dataset, set `--dataset [mnist|adult|covertype|MagicTelescope|mushroom|statlog]`.

## Examples

```
python3 main.py --model 'PlugTS' --dataset 'mnist'
```

```
python3 main.py --model 'PlugTS_P' --dataset 'mnist'
```