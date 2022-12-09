Parallel k-Nearest-Neighbour implementation in CUDA C/C++
=========================================

Uses the following code:

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
After starting the program, you are asked to specify the dataset (e.g. `datasets/iris_dataset/iris.data.csv`).

## Modes of knn program
The mode can be defined by setting either `#define EVALUATE 0` or `#define EVALUATE 1` in `knn.c`.
1) User mode (`#define EVALUATE 0`): User can specify a query point and the parameter k. The program then classifies this point.
2) Evaluation mode (`#define EVALUATE 1`): Determine the accuracy of the knn model using leave-one-out-cross validation (LOOCV). LOOCV is performed for all odd $k$ between 1 and the number of points in the dataset.