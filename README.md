Parallel k-Nearest-Neighbour implementation in CUDA C/C++
=========================================

Uses the following code:
- Based on the sequential knn code by [cdilga](https://github.com/cdilga/knn-c).
- `greatest.h` for unit testing. Check it out [on github](https://github.com/silentbicycle/greatest).
- Test Dataset sourced from
    - UCI Machine Learning Repository - [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
    - UCI Machine Learning Repository - [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
    - INMET (National Meteorological Institute - Brazil) - [Climate Weather Surface of Brazil - Hourly](https://www.kaggle.com/datasets/PROPPG-PPG/hourly-weather-surface-brazil-southeast-region?resource=download)

## Compile and run code
```console
make all # Compile knn and test code
./bin/knn # Run knn code
./bin/test # Run test code
```
After starting the program, you are asked to specify the dataset (e.g. `datasets/iris_dataset/iris.data.csv`). You can delete all build files with:
```console
make clean
```

The code is developed and tested only on Linux.

## Switch between sequential code and parallel CUDA version
By setting `#define CUDA 1` in `knn.cu` you can define that the parallel CUDA version of the knn algorithm should be used. When you set `#define CUDA 0` the sequential version is used.

## Modes of CUDA knn program
The mode can be defined by setting either `#define EVALUATE 0` or `#define EVALUATE 1` in `knn.cu`.
1) User mode (`#define EVALUATE 0`): User can specify a query point and the parameter k. The program then classifies this point.
2) Evaluation mode (`#define EVALUATE 1`): Determine the accuracy of the knn model using leave-one-out-cross validation (LOOCV). LOOCV is performed for all odd $k$ between 1 and the number of points in the dataset.

## Input CSV files
The datasets must be stored as CSV files. They must not have a header. The last column must contain the class of each sample. All other columns must only contain numerical data. The python script `datasets/weather_dataset/prepare_weather_dataset_for_knn.py` can be used to prepare the weather dataset for the use with this knn algorithm (e.g. by removing non-numeric columns).

## Profiling
TBD @Chris

## Debugging
For debugging with small datasets it is useful to set `#define DEBUG 1` in `knn.cu`. Then you get verbose output. Furthermore, you can use the debugger [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html) via
```console
printf "set cuda memcheck on\nset cuda api_failures stop\ncatch throw\nr\nbt\ninfo locals\nthread 1\nbt\n" > ./tmp.txt
cuda-gdb -batch -x tmp.txt --args ./bin/knn
```
or
```console
cuda-gdb ./bin/knn
```