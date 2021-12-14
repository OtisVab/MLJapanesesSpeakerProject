# Japanesespeaker_ML
Repo for the Japanese Speaker Identification project as part of the Machine Learning group project

# Running

## Pytorch scripts
To run `train.py` script, you can use `--help` to get the list of arguments for the script, and then pass them as required. The script has two required arguments which are the dataset to use and the config file of the model to test, the rest has default values and is therefore not mandatory. E.g:

```
python train.py --basefile ../Data_SI/ae --epochs 100 --model_config configs/random.json
```

### Plotting

To plot the loss and accuracy you can use the `--plot` flag as follows

```
python train.py --basefile ../Data_SI/ae --epochs 100 --model_config configs/random.json --plot
```
