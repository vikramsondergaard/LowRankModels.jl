# Proxy-Eliminating Fair Data Projection

Proxy-Eliminating Fair Data Projection is a Julia package for applying various notions of fairness to unsupervised learning, projecting data to a fair space for use in fair classification tasks.

To run a test, use the following command (in the root directory of the repository, where this README resides):
```bash
julia --project=. test/fair_glrms/experiments/test_<fairness>.jl data stat seed [-k K] [-s S] [-a A] [-w W] [-x X] [-y Y]
```
A description of each parameter is given in the table below:

|Parameter Flag|Parameter Name|Description|Possible Value|
|--------------|--------------|-----------|--------------|
|    N/A       |    `data`    |The data set to be projected using PEFDP.|`adult`, `adult_test`, `celeba`, `celeba_test`|
|    N/A       |    `stat`    | The measure of statistical dependence to use.|`orthog`, `softorthog`, `hsic`, `nothing`|
|    N/A       |     `seed`   | The random seed with which to partition the data into train-test splits.|An integer between 0 and 9 inclusive.|
|    `-k`      |     `k`      | The rank of the low-rank matrices *A* and *B*.|Any natural number greater than 0.|
|    `-s`      |   `scales`   | Values of the trade-off parameter λ. Multiple values results in multiple projections, one with each λ. | A list of decimal numbers between 0 and 1 (inclusive).|
|    `-a`      |   `alpha`    | The value of α to use for fair GLRMS.|Any positive real number.|
|    `-w`      |   `weights`  | The weights to use for fair GLRMS. | A list of positive numbers, whose length is equal to the number of protected groups in the data, such that the sum of all these numbers is 1.|
|    `-x`      |   `startX`   | A starting matrix for *A*. | An *m* × *k* matrix of real numbers.|
|    `-y`      |   `startY`   | A starting matrix for *B*. | An *n* × *k* matrix of real numbers.|

## Setting Up

This work requires access to a GPU. I have tested this work on the ANU School of Computing (SoCo) cluster, and will provide extra support for this infrastructure. Please feel free to reach out at `<my first name>.<my last name>@anu.edu.au` if you need any support.

To access a shell with access to a GPU on the ANU SoCo cluster, please type in the following commands once you're in the repository:

```bash
srun -p gpu --mem=32G --ntasks 1 --cpus-per-task 24 -t 5-00:00 --gres=gpu:1 --pty bash
singularity shell --nv /opt/apps/containers/julia/julia-1.9.4-bookworm.sif
```

Then, navigate to the root directory of this repository (where this README is located) and call
```bash
julia --project=.
```
This will open a Julia REPL with this repository defined as the "project". To download all the required libraries for this code, please run
```julia
]instantiate
```
The `]` key opens up `Pkg`, Julia's package manager.

Once you have installed all the necessary libraries, please run, in the Julia REPL,
```julia
using LowRankModels
```
This will pre-compile all the code. If you run into compile-time errors when running this program later, try running this command again in the Julia REPL.

## Generating Data

To generate the Adult Income data set splits, you can run the script `data/adult/adult_dataset.py`. Because this file requires Python, you will need to quit the Julia Singularity environment, by pressing `Ctrl+D`. You can still keep access to the GPU, so please **don't press** `Ctrl+D` **twice!** To get access to a Singularity Python shell, you can run
```bash
singularity shell --nv /opt/apps/containers/pytorch-latest-py3.sif
```

Then, navigate to `data/adult` and run
```bash
python3 adult_dataset.py
```
to create all the training and test splits.

## Running PEFDP

One quick and easy example you can run to verify everything works is
```bash
julia --project=. test/fair_glrms/experiments/test_independence.jl adult hsic 0 -k 2 -s 0.6
```
The output to the console is quite verbose, so it shouldn't be too hard to see if everything is working!

If you get stuck at any point, you can recover the last iteration of *A* and *B* by using the automatically saved `checkpoint_X.csv` and `checkpoint_Y.csv` in the `startX` and `startY` parameters.