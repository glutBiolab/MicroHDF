# MicroHDF


# Code

### The code includes MicroHDF and benchmarks.

- Benchmark includes ```Random Forest, SVM, LASSO, MLPNN, CNN1D```，```DeepForest``` models.

- Install required packages,then install tensorflow

>tensorflow = 1.14.2
>
>python = 3.6
>
>scikit-learn = 1.0.2
>
>pandas = 1.2.3
>
>joblib = 1.2.0
>
>deep-forest = 0.1.7

Run Benchmarks, printing out its usage.

```
train_baseline.py
```

you will set data in ```config.py```

### MicroHDF

#### Import conda environment

```
conda env create -f MicroHDF.yaml
```

#### Run MicroDF

- If only abundance information is used, use

```
main_MicroHDF.py
```

- Suppose that we want to  Reproducing the experiments described in our paper

````
demo_multimade.py
````

