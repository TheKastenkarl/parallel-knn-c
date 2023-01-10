<<<<<<< HEAD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <float.h>
#include <time.h>

#include "greatest.h"
#include "terminal_user_input.h"
#include "nvtx3.hpp"

#define EVALUATE 0  // Choose between user mode (0) and k evaluation mode (1)
#define CUDA 1      // Choose between parallel/CUDA mode (1) and sequential mode (0)
#define DEBUG 1     // Define the debug level. Outputs verbose output if enabled (1) and not if disabled (0)
#define TIMER 1     // Define whether you want to measure and print the execution time of certain functions (1) or not (0)
#define MAJORITY_CLASS_PARALLEL 0 // Choose if the majority class of the k nearest neighbors is determined in parallel (1) or sequentially (0)

#define TPB_LOCAL_KNN_X 128 // Threads per block for calculating the local k nearest neighbors (x-dim: Number of data points)
#define TPB_LOCAL_KNN_Y 8   // Threads per block for calculating the local k nearest neighbors (y-dim: Number of query points); Requirement: TPB_LOCAL_KNN_X * TPB_LOCAL_KNN_Y <= 1024
#define TPB_GLOBAL_KNN 64   // Threads per block for calculating the global k nearest neighbors and determining the class (Note: max possible k = 93 with a value of 64 due to max. shared memory size)

#define SEED 10     // Seed to allow better comparison between different runs during which randomness is involved

// Define a testing suite that is external to reduce code in this file
SUITE_EXTERN(external_suite);

// Datatype allows classifications to be stored very efficiently
// Is an array of char*, which is a double char*
// In order to use this struct, you must first define an array of char* on the class
typedef struct {
  my_string* categories;
  int num_categories;
} Classifier_List;

/**
 * Class for storing and managing datasets. Can be used both on GPU and CPU memory.
 */
class Dataset {
public:
  int num_dimensions; // Number of dimensions of a single point, e.g. 2 when you have x- and y-dimensions
  int num_points;     // Number of data points in the dataset
  float* points;      // Array of size num_points * num_dimensions storing the data points
  int* categories;    // Array of size num_points storing the categories of the data points

  // Constructor. Member arrays do not get allocated.
  __host__ __device__ Dataset(const int num_dimensions, const int num_points) {
    this->num_dimensions = num_dimensions;
    this->num_points = num_points;
    this->points = nullptr;
    this->categories = nullptr;
  }

  // Constructor. Read dataset from a csv file. Member arrays get allocated.
  __host__ Dataset(const my_string filename, Classifier_List* class_list) {
    this->read_dataset_file(filename, class_list);
  }

  // Read dataset from a csv file. Member arrays get allocated.
  __host__ void read_dataset_file(const my_string filename, Classifier_List* class_list);

  // Set or change a data point of the dataset. Corresponding array entries in member variable
  // 'points' have to be allocated before calling this method.
  __host__ __device__ void set_point(const int point_id, float* point) {
    float* point_ = this->get_point(point_id);
    for (int i = 0; i < this->num_dimensions; ++i) {
      point_[i] = point[i];
    }
  }

  // Parse a point from a string and add it to the dataset. Corresponding array entries in member
  // variable 'points' have to be allocated before calling this method.
  __host__ void parse_point(const int point_id, const my_string line, Classifier_List* class_list);

  // Returns a pointer to the point_id'th point of the dataset.
  __host__ __device__ float* get_point(const int point_id) {
    return &(this->points[point_id * this->num_dimensions]);
  }
};

/**
 * Class for storing and managing the query points. Can be used both on GPU and CPU memory.
 */
class Query_Points {
private:
  bool storedOnGPU;          // Stores if instance is stored in GPU (true) or CPU memory (false)

public:
  int num_dimensions;        // Number of dimensions of a single point, e.g. 2 when you have x- and y-dimensions
  int num_points;            // Number of query points stored in the instance
  float* points;             // Array of size num_points * num_dimensions storing the data points
  int* neighbor_ids;         // Array of size this->num_points * k storing the dataset ids of the k nearest neighbors for each query point
  float* neighbor_distances; // Array of size this->num_points * k storing the distances to the k nearest neighbors for each query point
  int* qpoint_categories;    // Array of size this->num_points storing the determined classes of the query points

  // Constructor. Member arrays only get allocated if 'storedOnGPU' is true.
  __host__ __device__ Query_Points(const bool storedOnGPU, const int num_dimensions, const int num_points, const int k) {
    this->storedOnGPU = storedOnGPU;
    this->num_dimensions = num_dimensions;
    this->num_points = num_points;
    if (!this->storedOnGPU) {
      this->points = (float*) malloc(this->num_points * this->num_dimensions * sizeof(float));
      this->neighbor_ids = (int*) malloc(this->num_points * k * sizeof(int));
      this->neighbor_distances = (float*) malloc(this->num_points * k * sizeof(float));
      this->qpoint_categories = (int*) malloc(this->num_points * sizeof(int));
    } else {
      this->points = nullptr;
      this->neighbor_ids = nullptr;
      this->neighbor_distances = nullptr;
      this->qpoint_categories = nullptr;
    }
  }

  // Set or change a query point. Corresponding array entries in member variable
  // 'points' have to be allocated before calling this method.
  __host__ __device__ void set_point(const int qpoint_id, float* query_point) {
    float* point = this->get_point(qpoint_id);
    for (int i = 0; i < this->num_dimensions; ++i) {
      point[i] = query_point[i];
    }
  }

  // Read a query point from the command line (for manual = true) or generate it randomly (for manual = false)
  // and add it to the 'points' array at the position 'qpoint_id'. Corresponding array entries in member
  // variable 'points' have to be allocated before calling this method.
  __host__ void read_query_point_user(const int qpoint_id, bool manual) {
    float* point = this->get_point(qpoint_id);

    if (manual) {
      for (int i = 0; i < this->num_dimensions; ++i) {
      printf("%dth dimension: ", i);
      point[i] = read_float("");
      }
    } else {
      for (int i = 0; i < this->num_dimensions; ++i) { 
      point[i] = rand() % 25;
      #if DEBUG
      printf("Query point ID %d: %dth dimension value is: %f", qpoint_id, i, point[i]);
      #endif
      }
    }
    
    
  }

  // Returns a pointer to the point_id'th query point.
  __host__ __device__ float* get_point(const int point_id) {
    return &(this->points[point_id * this->num_dimensions]);
  }
};

// Calculate the euclidean distance / L2-norm between two points.
__host__ __device__ float point_distance(float* x, float* y, int dimensionality) {
  float dist = 0;

  float sum = 0;

  //for each element in each, get the squared difference
  for (int i = 0; i < dimensionality; i++) {
    //sum this squared difference
    sum = sum + pow(x[i] - y[i], 2);
  }

  //get this sum and find the square root
  dist = sqrt(sum);

  return dist;
}

// Compare two integers.
// If v1 is greater than v2, positive.
// If v1 is less than v2, negative.
// If equal, 0.
int compare_int(const void* v1, const void* v2) {
  int n1 = *(int*) v1;
  int n2 = *(int*) v2;
  if (n1 - n2 > 1) {
    return 1;
  } else if (n1 - n2 < -1) {
    return -1;
  }
  return n1 - n2;
}

// Find the most frequent element in a C array of length 'len' (https://www.geeksforgeeks.org/frequent-element-array/).
__host__ __device__ int most_frequent(int const* arr, const int len) {
  int maxcount = 0;
  int element_having_max_freq = -1;
  for (int i = 0; i < len; ++i) {
    int count = 0;
    for (int j = 0; j < len; ++j) {
      if (arr[i] == arr[j]) {
        ++count;
      }
    }
    if (count > maxcount) {
      maxcount = count;
      element_having_max_freq = arr[i];
    }
  }
  return element_having_max_freq;
}

/**
 * Doing a sequential k nearest neighbor search.
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points.
 * @param dataset Dataset used to determine the classes of the query points.
 * @return Array containing the determined categories of the query points. The result is also stored in 'query_points->qpoint_categories'.
 */
int* knn_search(const int k, Query_Points* query_points, Dataset* dataset) {
  for (int j = 0; j < query_points->num_points; ++j) {
    // Warn if k is even
    if (k % 2 == 0) {
      printf("[WARN] Warning: %d is even. Tie cases have undefined behaviour\n", k);
    }

    #if DEBUG
    printf("[DEBUG] k: %d\n", k);
    #endif

    // For the first k points, just add some values into the array. This way we can just update the largest in every step.
    for (int id = 0; id < k; ++id) {
      float distance = point_distance(query_points->get_point(0), dataset->get_point(id), dataset->num_dimensions);
      query_points->neighbor_distances[id] = distance;
      query_points->neighbor_ids[id] = id;
    }

    // Get the euclidean distance to every neighbor
    for (int id = k; id < dataset->num_points; ++id) {
      float distance = point_distance(query_points->get_point(0), dataset->get_point(id), dataset->num_dimensions);

      #if DEBUG
      printf("[DEBUG] Point distance: %.4f\n", distance);
      #endif

      // If the data point is closer to the query point than the neighbor with the largest distance, replace this
      // neighbor with the data point.
      float max = 0.0;
      int update_index = 0;
      // Determine the neighbor with the largest distance which is currently marked as one of the (preliminary) k nearest neighbors
      for (int j = 0; j < k; ++j) {
        if (query_points->neighbor_distances[j] > max) {
          max = query_points->neighbor_distances[j];
          update_index = j;
        }
        #if DEBUG
        printf("[DEBUG] Distance[%d]: %.4f\n", j, query_points->neighbor_distances[j]);
        #endif
      }
      #if DEBUG
      printf("[DEBUG] update_index max distance identified to be: %d at distance: %.4f\n", update_index, query_points->neighbor_distances[update_index]);
      #endif

      // If the current point distance is less than the largest recorded distance, replace the corresponding
      // neighbor with the new data point.
      if (query_points->neighbor_distances[update_index] > distance) {
        // Update the distance at update_index
        #if DEBUG
        printf("[DEBUG] Compare neighbour[%d] = %.4f\n", update_index, distance);
        #endif
        query_points->neighbor_distances[update_index] = distance;
        query_points->neighbor_ids[update_index] = id; 

        #if DEBUG
        printf("[DEBUG] category of new point: %d\n", dataset->categories[id]);
        #endif
      }
      #if DEBUG
      printf("============================================\n");
      #endif
    }
    // Find the most frequent category of the global k nearest neighbors
    // First, get all the categories of the global k nearest neighbors and put them into an array
    int neighbour_categories[k];

    for (int c = 0; c < k; c++) {
      int point_id = query_points->neighbor_ids[c];
      neighbour_categories[c] = dataset->categories[point_id];

      #if DEBUG
      printf("[DEBUG] query_points->neighbour[%d].distance: %.4f\n", c, query_points->neighbor_distances[c]);
      printf("[DEBUG] Category[%d]: %d\n", c, neighbour_categories[c]);
      #endif
    }

    // Second, find the most frequent category
    query_points->qpoint_categories[j] = most_frequent(neighbour_categories, k);
    #if DEBUG
    printf("[DEBUG] Determined class for query point %d: %d\n", j, query_points->qpoint_categories[j]);
    #endif
  }
  return query_points->qpoint_categories;
}

/**
 * Calculate the local k nearest neighbors (local knns) of multiple query points
 * given the dataset. Each thread handles one pair of a query and a query point.
 *
 * Local k nearest neighbors means that for every thread block (x-dim. of the blocks) you
 * consider a subset of the dataset and then you calculate for every thread block the k
 * nearest neighbors in this subset. Hence, these nearest neighbors are not global but
 * only local. You do this for all qeury points (y-dim. of the thread blocks).
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points.
 * @param dataset Dataset containing the data points with the potential k nearest neighbors.
 * @param local_knns_distances Array in which the result (distances to local knns) is stored. Must be 
 * allocated before calling the function and must have size (k * num_query_points * gridDim.x). Indices 
 * 0,...,(k-1) store the local knns for the first thread block of the first query point, indices 
 * k,...,(2k-1) store the local knns for the second thread block of the first query point, ... . After 
 * all local knns for one query point the local knns of the next query point are stored. The local knns of
 * each block are sorted in ascending order, e.g. index 0 stores the nearest neighbor of thread block 0.
 * @param local_knns_ids Array in which the result (dataset IDs of local knns) is stored. The rest is the
 * same as for 'local_knns_distances'.
 */
__global__ void calculate_local_knns(const int k, Query_Points* query_points, Dataset* dataset, float local_knns_distances[], int local_knns_ids[]) {
  const int id_datapoint = blockIdx.x * blockDim.x + threadIdx.x; // ID of the data point handled in the current thread
  const int id_qpoint = blockIdx.y * blockDim.y + threadIdx.y;    // ID of the query point handled in the current thread

  // Handle invalid data and query points
  if (id_qpoint >= query_points->num_points) return;
  bool valid_datapoint = true;
  if ((id_datapoint >= dataset->num_points)) {
    valid_datapoint = false;
  }

  // Setup shared memory for storing the calculated distances
  // smem_size = blockDim.x * blockDim.y * 4 byte
  // Max. shared memory size of 48KB is no problem here because blockDim.x * blockDim.y <= 1024
  extern __shared__ float s_distances[];

  // Calculate distances (valid data points have the actual distance, invalid data points have the
  // max. possible distance). Including invalid points with the max. distance simplifies following
  // calculations because else the thread blocks for which we calculate less than 'blockIdx.x'
  // distances would need a special treatment.
  float distance = FLT_MAX;
  if (valid_datapoint) {
    distance = point_distance(query_points->get_point(id_qpoint), dataset->get_point(id_datapoint), dataset->num_dimensions);
  }
  
  s_distances[threadIdx.x] = distance;
  __syncthreads();

  // Determine the local k nearest neighbors
  // If rank_distance=3, it means that the corresponding point is the 4th closest datapoint to the
  // query point of all dataset considered in this thread block.
  int rank_distance = 0;
  for (int i = 0; i < blockDim.x; ++i) {
    if (distance > s_distances[i]) {
        ++rank_distance;
    } else if ((distance == s_distances[i]) && (threadIdx.x > i)) {
        // Handle the case when 2 samples have the same distance
        // -> Only for one of the samples the rank should be increased
        // -> Here: Only increase the rank of the sample with the higher index should be increased
        ++rank_distance;
    }
  }

  #if DEBUG
  printf("[DEBUG] calculate_local_knns | id_datapoint = %d \t id_qpoint = %d \t rank_distance = %d \t\t distance = %.4f\n", id_datapoint, id_qpoint, rank_distance, distance);
  #endif

  // Check if the calculated distance in this thread belongs to the local k nearest neighbors and if yes,
  // add the corresponding point to the arrays storing the local k nearest neighbors
  if (rank_distance < k) {
    float* local_nn_distance = &(local_knns_distances[k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance]);
    *local_nn_distance = distance;
    int* local_nn_id = &(local_knns_ids[k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance]);
    *local_nn_id = id_datapoint;

    #if DEBUG
    printf("[DEBUG] calculate_local_knns | local_knns_distances[%d] = %.4f \t id_datapoint = %d \t category = %d \t id_qpoint = %d\n", k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance, distance, id_datapoint, dataset->categories[id_datapoint], id_qpoint);
    #endif
  }
}

/**
 * Calculate the global k nearest neighbors (global knns) of multiple query points given
 * their local k nearest neighbors (local knns). Each thread handles one query point.
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points. The result (indices and distances to global knns)
 * of this function is stored in 'query_points->neighbor_distances' and 'query_points->neighbor_ids'.
 * Arrays must be allocated before calling the function and must have size (k * query_points->num_points). Indices
 * 0,...,(k-1) store the global knns (distances & ids) for the first query point, indices k,...,(2k-1)
 * store the global knns for the second query point, and so on.
 * @param dataset Dataset containing the data points with the potential k nearest neighbors.
 * @param local_knns_distances Array containing the distances to the local knns.
 * @param local_knns_ids Array containing the dataset id of the local knns.
 * @param gridDimX_local_knn Grid dimension in x-direction (blockDim.x) of the function
 * 'calculate_local_knns()'.
 * @param block_offsets Array required only within the function. Must be allocated before
 * calling the function and must have size (query_points->num_points * gridDimX_local_knn). Can be freed
 * directly after calling the function.
 */
__global__ void calculate_global_knn(const int k, Query_Points* query_points, Dataset const* dataset, float local_knns_distances[], int local_knns_ids[], const int gridDimX_local_knn, int block_offsets[]) {
  const int id_qpoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_qpoint >= query_points->num_points) return;

  // Set all elements in block_offsets to zero (at least the ones corresponding to the query point handled in this thread)
  for (int j = 0; j < gridDimX_local_knn; ++j) {
    block_offsets[id_qpoint * gridDimX_local_knn + j] = 0;
  }

  // Find the global k nearest neighbors of from all the local k nearest neighbors.
  // local_knns_* stores all local k nearest neighbors sequentially, e.g. from index 0 to k-1 you have
  // the local k nearest neighbors of thread block 0, from index k to 2k-1 you have the local k nearest
  // neighbors of thread block 1, and so on. The local k nearest neighbors are sorted in ascending order,
  // e.g. index 0 stores the nearest neighbor of thread block 0.
  // Hence, to determine the global nearest neighbor we only have to check the first element of all
  // k nearest neighbors. Then, if we want to determine the (global) second nearest neighbor we have to do
  // the same but we have to consider that we have already used the first element of one of the local k
  // nearest neighbors. This is done by adding an individual offset value to every local k nearest
  // neighbors subarray. These offset values are stored in 'block_offsets'.
  for (int i = 0; i < k; ++i) {
    float min_distance = FLT_MAX; // initialize to max value
    int min_dist_block_id;
    for (int j = 0; j < gridDimX_local_knn; ++j) {
      int block_offset = block_offsets[id_qpoint * gridDimX_local_knn + j];
      if (block_offset >= 0) { // enforce that we can only use TPB_LOCAL_KNN elements from every local knn subarray
        float min_dist_block = local_knns_distances[k * gridDimX_local_knn * id_qpoint + k * j + block_offset]; // minimum unused distance of local knn subarray
        if (min_dist_block < min_distance) {
          min_distance = min_dist_block;
          min_dist_block_id = j;
        }
      }
    }

    int min_dist_block_offset = block_offsets[id_qpoint * gridDimX_local_knn + min_dist_block_id];
    int min_dist_index = k * gridDimX_local_knn * id_qpoint + min_dist_block_id * k + min_dist_block_offset;
    query_points->neighbor_ids[k * id_qpoint + i] = local_knns_ids[min_dist_index]; // = id_datapoint
    query_points->neighbor_distances[k * id_qpoint + i] = local_knns_distances[min_dist_index];
    if (min_dist_block_offset < (TPB_LOCAL_KNN_X - 1)) {
      block_offsets[id_qpoint * gridDimX_local_knn + min_dist_block_id] += 1;
    } else {
      // Handle the case k > TPB_LOCAL_KNN_X -> Then we need this to enforce that we can only
      // use TPB_LOCAL_KNN_X elements from every local knn subarray
      block_offsets[id_qpoint * gridDimX_local_knn + min_dist_block_id] = -1;
    }
    
    #if DEBUG
    int id_datapoint = query_points->neighbor_ids[k * id_qpoint + i];
    printf("[DEBUG] calculate_global_knn | q_points->n_distances[%d] = %.4f \t id_datapoint = %d \t category = %d \t id_qpoint = %d\n", k * id_qpoint + i, query_points->neighbor_distances[k * id_qpoint + i], id_datapoint, dataset->categories[id_datapoint], id_qpoint);
    #endif
  }
}

/**
 * Determine the classes of multiple query points given their global k nearest neighbors
 * (global knns) by determining the majority class of the global knns. Each thread handles one
 * query point.
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points and the indices and distances to the
 * k global nearest neighbors. The result, i.e. the classes of all query points, is stored in
 * 'query_points->qpoint_categories'. This array must be allocated before calling the function and
 * must have size 'query_points->num_points'. Index 0 stores the class of the first query point,
 * index 1 stores the class of the second query point, and so on.
 * @param dataset Dataset containing the data points.
 */
__global__ void determine_majority_classes_parallel(const int k, Query_Points* query_points, Dataset const* dataset) {
  const int id_qpoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_qpoint >= query_points->num_points) return;

  extern __shared__ int s_neighbor_categories[]; // size: k * blockDim.x * 4 byte (be careful because max. size of shared memory is 48KB)

  #if TIMER
  clock_t start, end;
  start = clock();
  double time_used;
  #endif

  for (int i = 0; i < k; ++i) {
    int point_id = query_points->neighbor_ids[id_qpoint * k + i];
    s_neighbor_categories[id_qpoint * k + i] = dataset->categories[point_id];
  }

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] determine_majority_classes_parallel - fill shared memory %f \n", time_used);

  start = clock();
  #endif

  query_points->qpoint_categories[id_qpoint] = most_frequent(s_neighbor_categories + k * id_qpoint, k);

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] determine_majority_classes_parallel -  %f \n", time_used);
  #endif

  #if DEBUG
  printf("[DEBUG] Predicted category = %d \t id_qpoint = %d\n", query_points->qpoint_categories[id_qpoint], id_qpoint);
  #endif
}

/**
 * Determine the classes of multiple query points given their global k nearest neighbors
 * (global knns) by determining the majority class of the global knns.
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points and the indices and distances to the
 * k global nearest neighbors. The result, i.e. the classes of all query points, is stored in
 * 'query_points->qpoint_categories'. This array must be allocated before calling the function and
 * must have size 'query_points->num_points'. Index 0 stores the class of the first query point,
 * index 1 stores the class of the second query point, and so on.
 * @param dataset Dataset containing the data points.
 */
__host__ void determine_majority_classes(const int k, Query_Points* query_points, Dataset const* dataset) {
  for (int i = 0; i < query_points->num_points; ++i) {
    // Store the categories of the global k nearest neighbors in an array
    int neighbor_categories[k];
    for (int c = 0; c < k; c++) {
      int point_id = query_points->neighbor_ids[i * k + c];
      neighbor_categories[c] = dataset->categories[point_id];
    }
    query_points->qpoint_categories[i] = most_frequent(neighbor_categories, k);
    #if DEBUG
    printf("[DEBUG] Predicted category = %d \t id_qpoint = %d\n", query_points->qpoint_categories[i], i);
    #endif
  }
}

// Print a point
__host__ __device__ void print_point(float const* point, const int category, const int num_dimensions) {
  printf("(");
  int i = 0;
  do {
    if (i > 0) {
      printf(", ");
    }
    printf("%.4f", point[i]);
    i++;
  } while(i < num_dimensions);
  printf(") %d\n", category);
}

// Print the coordinates and categories of all points of the dataset.
__host__ __device__ void print_dataset(Dataset* dataset) {
  printf("Dataset\nDimensionality: %d\nNumber of Points: %d\n", dataset->num_dimensions, dataset->num_points);
  for (int i = 0; i < dataset->num_points; ++i) {
    print_point(dataset->get_point(i), dataset->categories[i], dataset->num_dimensions);
  }
}

void print_classes(Classifier_List classes) {
  for (int i = 0; i < classes.num_categories; ++i) {
    printf("Categories: %s\n", classes.categories[i].str);
  }
}

/**
 * Print the coordinates and categories of all points of a dataset stored in GPU memory.
 * Can be called from within a global function. 
 *
 * @param dataset Dataset used to determine the classes of the query points.
 */
__global__ void print_dataset_parallel(Dataset* dataset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) return;
  print_dataset(dataset);
}

/**
 * Doing a parallel k nearest neighbor search using the GPU.
 * The knn search is parallelized both over query points and datasets points.
 *
 * @param k Number of neighbors to consider to determine the k nearest neighbors.
 * @param query_points Class instance storing the query points. After the execution of the function it
 * also stores information about the global k nearest neighbors.
 * @param dataset Dataset used to determine the classes of the query points.
 * @return Array containing the determined categories of the query points. The result is also stored in
 * 'query_points->qpoint_categories'.
 */
int* knn_search_parallel(const int k, Query_Points* query_points, Dataset const* dataset) {
  // Warn if k is even.
  if (k % 2 == 0) {
    printf("[WARN] Warning: k = %d is even. Tie cases have undefined behaviour!\n", k);
  }

  #if DEBUG
  printf("[DEBUG] k = %d\n", k);
  for (int i = 0; i < query_points->num_points; ++i) {
    printf("[DEBUG] Query point %d: ", i);
    print_point(query_points->get_point(i), -1, query_points->num_dimensions);
  }
  #endif

  // Declare pointers pointing to GPU memory
  Query_Points* query_points_device;
  Dataset* dataset_device;
  float* local_knns_distances_device;
  int* local_knns_ids_device;
  int* block_offsets;

  // GPU pointers required as helper variables for copying the data structures
  float* dataset_points_device;
  int* dataset_categories_device;

  float* query_points_points_device;
  int* query_points_neighbor_ids_device;
  float* query_points_neighbor_distances_device;
  int* query_points_qpoint_categories_device;

  // Allocate GPU memory
  // 1) Dataset
  cudaMalloc(&dataset_device, sizeof(Dataset));
  cudaMalloc(&dataset_points_device, dataset->num_points * dataset->num_dimensions * sizeof(float));
  cudaMalloc(&dataset_categories_device, dataset->num_points * sizeof(int));

  // 2) Query points
  cudaMalloc(&query_points_device, sizeof(Query_Points));
  cudaMalloc(&query_points_points_device, query_points->num_points * query_points->num_dimensions * sizeof(float));
  cudaMalloc(&query_points_neighbor_ids_device, query_points->num_points * k * sizeof(int));
  cudaMalloc(&query_points_neighbor_distances_device, query_points->num_points * k * sizeof(float));
  cudaMalloc(&query_points_qpoint_categories_device, query_points->num_points * sizeof(int));

  // Copy memory to GPU
  // 1) Dataset
  cudaMemcpy(dataset_device, dataset, sizeof(Dataset), cudaMemcpyHostToDevice);
  cudaMemcpy(dataset_points_device, dataset->points, dataset->num_points * dataset->num_dimensions * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dataset_device->points), &dataset_points_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(dataset_categories_device, dataset->categories, dataset->num_points * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dataset_device->categories), &dataset_categories_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct

  // 2) Query points
  cudaMemcpy(query_points_device, query_points, sizeof(Query_Points), cudaMemcpyHostToDevice);
  cudaMemcpy(query_points_points_device, query_points->points, query_points->num_points * query_points->num_dimensions * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(query_points_device->points), &query_points_points_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->neighbor_ids), &query_points_neighbor_ids_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->neighbor_distances), &query_points_neighbor_distances_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->qpoint_categories), &query_points_qpoint_categories_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct

  #if DEBUG
  printf("[DEBUG] Dataset used for knn search:\n");
  print_dataset_parallel<<<1, 32>>>(dataset_device);
  #endif

  // Initialize the grid and block dimensions and calculate the local k nearest neighbors
  // Use 2D block and thread grid: x - dimension of the dataset points, y - dimension of the query points
  const int gridDimY_local_knn = (query_points->num_points + TPB_LOCAL_KNN_Y - 1) / TPB_LOCAL_KNN_Y;
  const int gridDimX_local_knn = (dataset->num_points + TPB_LOCAL_KNN_X - 1) / TPB_LOCAL_KNN_X;
  int smem_size = TPB_LOCAL_KNN_X * TPB_LOCAL_KNN_Y * sizeof(float);
  cudaMalloc(&local_knns_distances_device, query_points->num_points * k * gridDimX_local_knn * sizeof(float));
  cudaMalloc(&local_knns_ids_device, query_points->num_points * k * gridDimX_local_knn * sizeof(int));
  calculate_local_knns<<<dim3(gridDimX_local_knn, gridDimY_local_knn, 1), dim3(TPB_LOCAL_KNN_X, TPB_LOCAL_KNN_Y, 1), smem_size>>>(k, query_points_device, dataset_device, local_knns_distances_device, local_knns_ids_device);
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions and calculate the global k nearest neighbors
  cudaMalloc(&block_offsets, query_points->num_points * gridDimX_local_knn * sizeof(int));
  const int num_blocks_global_knn = (query_points->num_points + TPB_GLOBAL_KNN - 1) / TPB_GLOBAL_KNN;
  calculate_global_knn<<<num_blocks_global_knn, TPB_GLOBAL_KNN>>>(k, query_points_device, dataset_device, local_knns_distances_device, local_knns_ids_device, gridDimX_local_knn, block_offsets);
  cudaFree(block_offsets);
  cudaDeviceSynchronize();

  // Determine the category of the query points by determining the majority class of the global k nearest neighbors
  // (in parallel or sequentially). And copy the result from GPU memory to the CPU memory.
  #if MAJORITY_CLASS_PARALLEL
  smem_size = k * TPB_GLOBAL_KNN * sizeof(int);
  nvtxRangePush("knn_search_parallel - determine_majority_classes_parallel (parallel)");
  determine_majority_classes_parallel<<<num_blocks_global_knn, TPB_GLOBAL_KNN, smem_size>>>(k, query_points_device, dataset_device);
  nvtxRangePop();
  cudaMemcpy(query_points->neighbor_ids, query_points_neighbor_ids_device, k * query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(query_points->qpoint_categories, query_points_qpoint_categories_device, query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  #else
  cudaMemcpy(query_points->neighbor_ids, query_points_neighbor_ids_device, k * query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  nvtxRangePush("knn_search_parallel - determine_majority_classes (sequential)");
  determine_majority_classes(k, query_points, dataset);
  nvtxRangePop();
  #endif

  // Free the GPU memory
  cudaFree(query_points_device);
  cudaFree(query_points_points_device);
  cudaFree(query_points_neighbor_ids_device);
  cudaFree(query_points_neighbor_distances_device);
  cudaFree(query_points_qpoint_categories_device);

  cudaFree(dataset_device);
  cudaFree(dataset_points_device);
  cudaFree(dataset_categories_device);

  cudaFree(local_knns_distances_device);
  cudaFree(local_knns_ids_device);

  return query_points->qpoint_categories;
}

// Function that takes in a classification integer and returns a classification string.
// Requires a map between the integers and the string in the form of a Classifier_List datatype.
my_string classify(const Classifier_List category_map, const int category) {
  my_string class_string = category_map.categories[category];
  return class_string;
}

// Count the number of columns of a csv file.
int count_fields(char const* buffer) {
  int count = 1;
  int pos = 0;
  char current;
  do {
    current = buffer[pos];
    if (current == ',') {
      count++;
    }
    pos++;
  } while(current != '\n' && current != '\0');
  #if DEBUG
  printf("[DEBUG] Number of fields: %d\n", count);
  #endif
  return count;
}

// Function that takes in a classification string and returns a classification integer.
// Requires a map between the integers and the string in the form of a Classifier_List datatype.
int get_class_num(const my_string in_string, Classifier_List* class_list) {
  // Could improve with a Levenshtein Distance calculation to account for human errors.
  // Also, if i is zero, we won't even need to check if it's in there, we know it's not.
  #if DEBUG
  printf("[DEBUG] class_list->num_categories: %d\n", class_list->num_categories);
  #endif

  for (int i = 0; i < class_list->num_categories; i++) {
    if (strcmp(class_list->categories[i].str, in_string.str) == 0) {
      return i;
    }
  }
  // If it isn't present in the existing array, we need to add it in.
  // Increment the count of categories.
  class_list->num_categories++;
  #if DEBUG
  printf("[DEBUG] Class list categories: %d\n", class_list->num_categories);
  #endif
  class_list->categories = (my_string*) realloc(class_list->categories, sizeof(my_string) * class_list->num_categories);
  class_list->categories[class_list->num_categories - 1] = in_string;
  return class_list->num_categories - 1;
}

// Function to read lines from csv. Takes a file name.
my_string extract_field(my_string line, const int field) {
  my_string return_value;
  // Using https://support.microsoft.com/en-us/help/51327/info-strtok-c-function----documentation-supplement
  if (field > count_fields(line.str)) {
    strcpy(return_value.str, "\0");
    return return_value;
  }
  // Potentially unsafe
  char *token;

  token = strtok(line.str, " ,");
  // Call strtok "field" times.
  // Return that value of the token.
  for (int i = 1; i < field; i++) {
    #if DEBUG
    printf("[DEBUG] Token is: %s\n", token);
    #endif

    token = strtok(NULL, " ,");
    #if DEBUG
    printf("[DEBUG] Before copy in loop\n");
    #endif
  }
  strncpy(return_value.str, token, sizeof(return_value.str));

  return return_value;
}

// Count the number of rows of a csv file.
int count_lines(const my_string filename) {
  FILE *file;
  if (access(filename.str, F_OK) == -1) {
    printf("[ERROR] Could not find file");
    return -1;
  }
  file = fopen(filename.str, "r");
  char buffer[1024];
  int count = 0;
  while(fgets(buffer, 1024, file)) {
    // True if line is non-empty (only \n means also non-empty)
    count++;
  }
  fclose(file);
  #if DEBUG
  printf("[DEBUG] Number of lines in input file: %d\n", count);
  #endif
  return count;
}

// Parse a point from a string and add it to the dataset. Corresponding array entries in member
// variable 'points' have to be allocated before calling this method.
void Dataset::parse_point(const int point_id, const my_string line, Classifier_List* class_list) {
  float* point = this->get_point(point_id);
  for (int i = 0; i < this->num_dimensions; ++i) {
    // Go through and pull out the first num fields and construct a point out of them.
    // Pass the string into a function that just mocks out and returns 1
    // since the extract_field function extracts with a base 1, rather than base of 0.
    point[i] = atof(extract_field(line, i + 1).str);
  }

  // The data for the class is in the last column
  this->categories[point_id] = get_class_num(extract_field(line, num_dimensions + 1), class_list);
  #if DEBUG
  print_point(point, this->categories[point_id], this->num_dimensions);
  #endif
}

// Read dataset from a csv file. Member arrays get allocated.
void Dataset::read_dataset_file(const my_string filename, Classifier_List* class_list) {
  // Read the number of lines in before opening the files
  this->num_points = count_lines(filename);
  
  // From that, it should return some struct
  FILE *file;
  if (access(filename.str, F_OK) == -1) {
    printf("[ERROR] Could not find file.");
  }
  file = fopen(filename.str, "r");

  // Struct should contain a 2d array with the lines, in each with data separated into array elements
  char *buffer;
  buffer = (char*) malloc(1024 * sizeof(char));
  (void)! fscanf(file, "%s\n", buffer);

  // Count the commas
  this->num_dimensions = count_fields(buffer) - 1;

  // Allocate memory
  this->points = (float*) malloc(this->num_points * this->num_dimensions * sizeof(float));
  this->categories = (int*) malloc(this->num_points * sizeof(int));

  my_string buffer_string;
  strcpy(buffer_string.str, buffer);

  int i = 0;
  // For each line, parse the point and add it to the dataset
  do {
    this->parse_point(i, buffer_string, class_list);

    ++i;
    // Don't do this on the last iteration of the loop
    if (!(i == this->num_points)) {
      (void)! fscanf(file, "%s\n", buffer);
      strcpy(buffer_string.str, buffer);
    }
  } while (i < this->num_points);

  // Now we can essentially read in the first "count" fields and cast to float
  // Read in the last field, IE count and add a class for the
  free(buffer);
}

// Create a new Classifier_List structure.
Classifier_List new_classifier_list() {
  int num_categories = 0;
  my_string *categories;
  categories = (my_string*) malloc(sizeof(my_string));
  Classifier_List new_list = {categories, num_categories};
  return new_list;
}

// Determine the accuracy of the knn model using leave-one-out-cross validation (LOOCV).
float evaluate_knn(const int k, Dataset* benchmark_dataset) {
  #if DEBUG
  printf("============================================\n");
  printf("[DEBUG] Complete dataset:\n");
  print_dataset(benchmark_dataset);
  #endif

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif

  Dataset comparison_dataset(benchmark_dataset->num_dimensions, benchmark_dataset->num_points - 1);
  comparison_dataset.points = (float*) malloc(comparison_dataset.num_points * comparison_dataset.num_dimensions * sizeof(float));
  comparison_dataset.categories = (int*) malloc(comparison_dataset.num_points * sizeof(int));

  int sum_correct = 0;
  // Make a copy of the dataset, except missing the i'th term.
  for (int i = 0; i < benchmark_dataset->num_points; i++) {
    
    // Loop through the dataset the number of times there are points
    #if DEBUG
    printf("============================================\n");
    printf("[DEBUG] i: %d\n", i);
    #endif
    for (int j = 0; j < comparison_dataset.num_points; j++) {
      // Don't copy the ith term.
      // Index will point to the correct term.
      int index;
      if (j >= i) {
        index = j + 1;
      } else {
        index = j;
      }
      #if DEBUG
      printf("[DEBUG] Index: %d\n", index);
      #endif

      // Copy point
      float* comparison_datapoint = comparison_dataset.get_point(j);
      float* benchmark_datapoint = benchmark_dataset->get_point(index);
      for (int dim = 0; dim < comparison_dataset.num_dimensions; ++dim) {
        comparison_datapoint[dim] = benchmark_datapoint[dim];
      }
      comparison_dataset.categories[j] = benchmark_dataset->categories[index];
    }

    // Create a query point out of that i'th term
    Query_Points query_point(false, benchmark_dataset->num_dimensions, 1, k);
    query_point.set_point(0, benchmark_dataset->get_point(i));
    #if DEBUG
    printf("[DEBUG] Gets to the knn search\n");
    #endif
    // If the classification matches the category, add it to the sum.
    #if CUDA
    int* qpoint_categories = knn_search_parallel(k, &query_point, &comparison_dataset);
    if (qpoint_categories[0] == benchmark_dataset->categories[i]) {
      sum_correct++;
    }
    #else
    int* qpoint_categories = knn_search(k, &query_point, &comparison_dataset);
    if ( qpoint_categories[0] == benchmark_dataset->categories[i]) {
      sum_correct++;
    }
    #endif
    #if DEBUG
    printf("[DEBUG] Actual category: %d\n", benchmark_dataset->categories[i]);
    #endif

    free(query_point.points);
    free(query_point.neighbor_ids);
    free(query_point.neighbor_distances);
    free(query_point.qpoint_categories);
  }

  float accuracy = (float) sum_correct / (float) benchmark_dataset->num_points;

  // Free CPU memory
  free(comparison_dataset.points);
  free(comparison_dataset.categories);

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] evaluate_knn - execution time for k = %d neigbors: %f sec\n", k, time_used);
  #endif

  return accuracy;
}

#ifndef NDEBUG
// Definitions required for the testrunner
GREATEST_MAIN_DEFS();
#endif

// This main function takes commandline arguments
int main (int argc, char** argv) {
  // Set seed for random point generation
  srand(SEED);

  // Wrapped in #ifndef so we can make a release version
  #ifndef NDEBUG
  // Setup required testing
    GREATEST_MAIN_BEGIN();

    // Runs tests from external file specified above
    RUN_SUITE(external_suite);

    // Show results of the testing
    GREATEST_MAIN_END();
  #endif

  // Read a dataset
  Classifier_List class_list = new_classifier_list();
  my_string filename = read_string("Filename: ");
  Dataset generic_dataset(filename, &class_list);

  int k = read_integer("Please put the desired number of neighbours k for the search: ");

  #if !EVALUATE

  int num_query_points = 1;
  num_query_points = read_integer("How many query points do you want to enter?: ");
  bool query_points_manually = read_boolean("Do you want to enter the query points manually? (yes/no) If no, the query points will be chosen randomly: ");
  
  Query_Points query_points(false, generic_dataset.num_dimensions, num_query_points, k);

  // loop to create the number of required query points
  for (int i = 0; i < num_query_points; ++i) {
    query_points.read_query_point_user(i, query_points_manually);
  }

  #if CUDA

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif

  
  nvtxRangePush("main - knn_search_parallel (parallel)");
  int* qpoint_categories = knn_search_parallel(k, &query_points, &generic_dataset);
  nvtxRangePop();

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] main - knn_search_parallel - execution time for %d k neigbours and %d query points: %f \n", k, num_query_points, time_used);
  #endif

  #else

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif
  nvtxRangePush("main - knn_search (sequential)");
  int* qpoint_categories = knn_search(k, &query_points, &generic_dataset);
  nvtxRangePop();
  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] main - knn_search - execution time for %d k neigbours and %d query points: %f \n", k, num_query_points, time_used);
  #endif

  #endif

  for (int j = 0; j < num_query_points; ++j) {
    my_string class_string = classify(class_list, qpoint_categories[j]);
    printf("Query point ID %d classified as: %s\n", j, class_string.str);
  }

  free(query_points.points);
  free(query_points.neighbor_ids);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);
  
  #endif

  #if EVALUATE
  for (int k = 1; k < generic_dataset.num_points; k = k + 2) {
    printf("k: %d, accuracy: %.4f\n", k, evaluate_knn(k, &generic_dataset));
    #if DEBUG
    printf("++++++++++++++++++++++++++++++++++++++++++++\n\n");
    #endif
  }
  // For values of k up to the number of points that exist in the dataset
  #endif

  // Free CPU memory
  free(class_list.categories);
  free(generic_dataset.points);
  free(generic_dataset.categories);

  return 0;
=======
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <float.h>
#include <time.h>

#include "greatest.h"
#include "terminal_user_input.h"
#include "nvtx3.hpp"

#define SEED 10     // seed to allow better comparison between different runs during wich randomness is involved

#define EVALUATE 0  // Choose between user mode (0) and k evaluation mode (1)
#define CUDA 1      // Choose between parallel/CUDA mode (1) and sequential mode (0)
#define DEBUG 0     // Define the debug level. Outputs verbose output if enabled (1) and not if disabled (0)
#define TIMER 0     // Define whether you want to measure and print the execution time of certain functions (1) or not (0)
#define MAJORITY_CLASS_PARALLEL 0 // Choose if the majority class of the k nearest neighbors is determined in parallel (1) or sequentially (0)

#define TPB_LOCAL_KNN_X 128 // Threads per block for calculating the local k nearest neighbors (x-dim: Number of dataset)
#define TPB_LOCAL_KNN_Y 8   // Threads per block for calculating the local k nearest neighbors (y-dim: Number of query points)
#define TPB_GLOBAL_KNN 64   // Threads per block for calculating the global k nearest neighbors and determining the class

//Define a testing suite that is external to reduce code in this file
SUITE_EXTERN(external_suite);

//Datatype allows classifications to be stored very efficiently
//Is an array of char *, which is a double char *
//In order to use this struct, you must first define an array of char* on the class
typedef struct {
  my_string *categories;
  int num_categories;
} Classifier_List;

class Dataset {
private:
  bool storedOnGPU; // Stores if instance is stored in GPU (true) or CPU memory (false)

public:
  int num_dimensions; // number of dimensions of a single point, e.g. 2 when you have x- and y-values
  int num_points;
  float* points; // array of size num_points * num_dimensions
  int* categories; // array of size num_points

  __host__ __device__ Dataset(const int num_dimensions, const int num_points) {
    this->num_dimensions = num_dimensions;
    this->num_points = num_points;
    this->points = nullptr;
    this->categories = nullptr;
  }

  __host__ Dataset(const my_string filename, Classifier_List* class_list) {
    this->read_dataset_file(filename, class_list);
  }

  __host__ void read_dataset_file(const my_string filename, Classifier_List* class_list);

  __host__ __device__ void set_point(const int point_id, float* point) {
    float* point_ = this->get_point(point_id);
    for (int i = 0; i < this->num_dimensions; ++i) {
      point_[i] = point[i];
    }
  }

  __host__ void parse_point(const int point_id, const my_string line, Classifier_List* class_list);

  __host__ __device__ float* get_point(const int point_id) {
    return &(this->points[point_id * this->num_dimensions]);
  }
};

class Query_Points {
private:
  bool storedOnGPU; // Stores if instance is stored in GPU (true) or CPU memory (false)

public:
  int num_dimensions; // number of dimensions of a single point, e.g. 2 when you have x- and y-values
  int num_points;
  float* points; // array of size num_points * num_dimensions
  int* neighbor_idx; // array of size this->num_points * k
  float* neighbor_distances; // array of size this->num_points * k
  int* qpoint_categories; // array of size this->num_points storing the determined classes of the query points

  __host__ __device__ Query_Points(const bool storedOnGPU, const int num_dimensions, const int num_points, const int k) {
    this->storedOnGPU = storedOnGPU;
    this->num_dimensions = num_dimensions;
    this->num_points = num_points;
    if (!this->storedOnGPU) {
      this->points = (float*) malloc(this->num_points * this->num_dimensions * sizeof(float));
      this->neighbor_idx = (int*) malloc(this->num_points * k * sizeof(int));
      this->neighbor_distances = (float*) malloc(this->num_points * k * sizeof(float));
      this->qpoint_categories = (int*) malloc(this->num_points * sizeof(int));
    }
  }

  __host__ __device__ void set_point(const int qpoint_id, float* query_point) {
    float* point = this->get_point(qpoint_id);
    for (int i = 0; i < this->num_dimensions; ++i) {
      point[i] = query_point[i];
    }
  }

  __host__ void read_query_point_user(const int qpoint_id, bool manual) {
    float* point = this->get_point(qpoint_id);

    if(manual){
      for (int i = 0; i < this->num_dimensions; ++i) {
      printf("%dQuery point ID %d: th dimension: ", qpoint_id, i);
      point[i] = read_float("");
      }
    }else{
      for (int i = 0; i < this->num_dimensions; ++i) { 
      point[i] = rand() % 25;
      #if DEBUG
      printf("Query point ID %d: %dth dimension value is: %f", qpoint_id, i, point[i]);
      #endif
      }
    }
    
    
  }

  __host__ __device__ float* get_point(const int point_id) {
    return &(this->points[point_id * this->num_dimensions]);
  }
};

//Distance
//Return: number with the distance, a float
//Inputs, euclidean point, x and euclidean point y
__host__ __device__ float point_distance(float* x, float* y, int dimensionality) {
  float dist = 0;

  float sum = 0;

  //for each element in each, get the squared difference
  for (int i = 0; i < dimensionality; i++) {
    //sum this squared difference
    sum = sum + pow(x[i] - y[i], 2);
  }

  //get this sum and find the square root
  dist = sqrt(sum);

  return dist;
}

//Compare two integers
int compare_int(const void *v1, const void *v2) {
  //if value 1 is greater than value 2, positive,
  //if equal, 0
  //if value 1 less value 2, negative

  int n1 = *(int*)v1;
  int n2 = *(int*)v2;
  if (n1 - n2 > 1) {
    return 1;
  } else if (n1 - n2 < -1) {
    return -1;
  }
  return n1 - n2;
}

// Find the most frequent element in a C array (https://www.geeksforgeeks.org/frequent-element-array/)
__host__ __device__ int most_frequent(int const* arr, const int n) {
  int maxcount = 0;
  int element_having_max_freq = -1;
  for (int i = 0; i < n; ++i) {
    int count = 0;
    for (int j = 0; j < n; ++j) {
      if (arr[i] == arr[j]) {
        ++count;
      }
    }
    if (count > maxcount) {
      maxcount = count;
      element_having_max_freq = arr[i];
    }
  }
  return element_having_max_freq;
}

//Doing a k nearest neighbour search
int* knn_search(const int k, Query_Points* query_points, Dataset* dataset) {
  for(int j = 0; j < query_points->num_points; ++j){
    // Warn if k is even
    if (k % 2 == 0) {
      printf("[WARN] Warning: %d is even. Tie cases have undefined behaviour\n", k);
    }

    #if DEBUG
    printf("[DEBUG] k: %d\n", k);
    #endif

    // For the first k points, just add whatever junk into the array. This way we can just update the largest.
    for (int id = 0; id < k; ++id) {
      float distance = point_distance(query_points->get_point(j), dataset->get_point(id), dataset->num_dimensions);
      query_points->neighbor_distances[id] = distance;
      query_points->neighbor_idx[id] = id;
    }

    // Get the euclidean distance to every neighbour,
    for (int id = k; id < dataset->num_points; ++id) {
      float distance = point_distance(query_points->get_point(j), dataset->get_point(id), dataset->num_dimensions);

      #if DEBUG
      printf("[DEBUG] Point distance: %.4f\n", distance);
      #endif

      // if the neighbour is closer than the last, or it's null pointer distance closest keep it in a distance array
      // loop through all of the values for k, and keep the value from the comparison list for the query_point which is update_index.
      float max = 0;
      int update_index = 0;
      for (int j = 0; j < k; j++) {
        if (query_points->neighbor_distances[j] > max) {
          max = query_points->neighbor_distances[j];
          update_index = j;
        }
        #if DEBUG
        printf("[DEBUG] Distance[%d]: %.4f\n", j, query_points->neighbor_distances[j]);
        #endif
      }
      #if DEBUG
      printf("[DEBUG] update_index max distance identified to be: %d at distance: %.4f\n", update_index, query_points->neighbor_distances[update_index]);
      #endif

      // if the current point distance is less than the largest recorded distance, or if the distances haven't been set
      if (query_points->neighbor_distances[update_index] > distance) {
        // Update the distance at update_index
        #if DEBUG
        printf("[DEBUG] Compare neighbour[%d] = %.4f\n", update_index, distance);
        #endif
        query_points->neighbor_distances[update_index] = distance;
        query_points->neighbor_idx[update_index] = id; 

        #if DEBUG
        printf("[DEBUG] category of new point: %d\n", dataset->categories[id]);
        #endif
      }
      #if DEBUG
      printf("============================================\n");
      #endif
    }
    // Find the most frequent category of the global k nearest neighbors
    // First, get all the categories of the global k nearest neighbors and put them into an array
    int neighbour_categories[k];

    for (int c = 0; c < k; c++) {
      int point_id = query_points->neighbor_idx[c];
      neighbour_categories[c] = dataset->categories[point_id];

      #if DEBUG
      printf("[DEBUG] query_point->neighbour[%d].distance: %.4f\n", c, query_point->neighbor_distances[c]);
      printf("[DEBUG] Category[%d]: %d\n", c, neighbour_categories[c]);
      #endif
    }

    // Second, find the most frequent category
    query_points->qpoint_categories[j] = most_frequent(neighbour_categories, k);
    #if DEBUG
    printf("[DEBUG] Determined class: %d\n", category);
    #endif
  }
  return query_points->qpoint_categories;
}

/**
 * Calculate the local k nearest neighbors (local knns) of multiple comparison points
 * given the dataset. Each thread handles one pair of a query and a comparison point.
 *
 * Local k nearest neighbors means that for every thread block (x-dim. of the blocks) you
 * consider a subset of the dataset and then you calculate for every thread block the k
 * nearest neighbors in this subset. Hence, these nearest neighbors are not global but
 * only local. You do this for all qeury points (y-dim. of the thread blocks).
 *
 * @param k Number of neighbor to consider to determine the k nearest neighbors.
 * @param cpoints Array containing the query points (= comparison points).
 * @param num_cpoints Number of comparison points.
 * @param dataset Dataset containing the potential k nearest neighbor.
 * @param local_knns Array in which the result (local knns) is stored. Must be allocated before 
 * calling the function and must have size (k * num_cpoints * blockDim.x). Indices 0,...,(k-1) store
 * the local knns for the first thread block of the first query point, indices k,...,(2k-1) store
 * the local knns for the second thread block of the first query point, ... . After all local
 * knns for one query point the local knns of the next query point are stored. The local knns of
 * each block are sorted in ascending order, e.g. index 0 stores the nearest neighbor of thread block 0.
 */
__global__ void calculate_local_knns(const int k, Query_Points* query_points, Dataset* dataset, float local_knns_distances[], int local_knns_idx[]) {
  const int id_datapoint = blockIdx.x * blockDim.x + threadIdx.x;
  const int id_qpoint = blockIdx.y * blockDim.y + threadIdx.y;
  if (id_qpoint >= query_points->num_points) return;
  bool valid_datapoint = true;
  if ((id_datapoint >= dataset->num_points)) {
    valid_datapoint = false;
  }

  extern __shared__ float s_distances[]; // smem_size = blockDim.x * blockDim.y * sizeof(float)

  // Calculate distances (valid points have the actual distance, invalid points have the max.
  // possible distance). Including invalid points with the max. distance simplifies following
  // calculations because else the thread blocks for which we calculate less than 'blockIdx.x'
  // distances would need a special treatment.
  float distance = FLT_MAX;
  if (valid_datapoint) {
    distance = point_distance(query_points->get_point(id_qpoint), dataset->get_point(id_datapoint), dataset->num_dimensions);
  }
  
  s_distances[threadIdx.x] = distance;
  __syncthreads();

  // Determine the local k nearest neighbors
  // If rank_distance=3, it means that the corresponding point is the 4th closest datapoint to the
  // query point of all dataset considered in this thread block.
  int rank_distance = 0;
  for (int i = 0; i < blockDim.x; ++i) {
    if (distance > s_distances[i]) {
        ++rank_distance;
    } else if ((distance == s_distances[i]) && (threadIdx.x > i)) {
        // Handle the case when 2 samples have the same distance
        // -> Only for one of the samples the rank should be increased
        // -> Here: Only increase the rank of the sample with the higher index should be increased
        ++rank_distance;
    }
  }

  #if DEBUG
  printf("[DEBUG] calculate_local_knns | id_datapoint = %d \t id_qpoint = %d \t rank_distance = %d \t\t distance = %.4f\n", id_datapoint, id_qpoint, rank_distance, distance);
  #endif

  // Check if the calculated distance in this thread belongs to the local k nearest neighbors and if yes,
  // add the corresponding point to the array storing the local k nearest neighbors
  if (rank_distance < k) {
    float* local_nn_distance = &(local_knns_distances[k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance]);
    *local_nn_distance = distance;
    int* local_nn_id = &(local_knns_idx[k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance]);
    *local_nn_id = id_datapoint;

    #if DEBUG
    printf("[DEBUG] calculate_local_knns | local_knns_distances[%d] = %.4f \t id_datapoint = %d \t category = %d \t id_qpoint = %d\n", k * gridDim.x * id_qpoint + k * blockIdx.x + rank_distance, distance, id_datapoint, dataset->categories[id_datapoint], id_qpoint);
    #endif
  }
}

/**
 * Calculate the global k nearest neighbors (global knns) of multiple comparison points given
 * their local k nearest neighbors (local knns). Each thread handles one comparison point.
 *
 * @param k Number of neighbor to consider to determine the k nearest neighbors.
 * @param num_cpoints Number of comparison points.
 * @param local_knns Array containing the local knns.
 * @param blockDimX_local_knn Block dimension in x-direction (blockDim.x) of the function
 * 'calculate_local_knns()'.
 * @param block_offsets Array required only within the function. Must be allocated before
 * calling the function and must have size (num_cpoints * blockDimX_local_knn). Can be freed
 * directly after calling the function.
 * @param global_knn Array in which the result (global knn) is stored. Must be allocated before
 * calling the function and must have size (k * num_cpoints). Indices 0,...,(k-1) store
 * the global knns for the first query point, indices k,...,(2k-1) store
 * the global knns for the second query point, and so on.
 */
__global__ void calculate_global_knn(const int k, Dataset const* dataset, Query_Points* query_points, float local_knns_distances[], int local_knns_idx[], const int blockDimX_local_knn, int block_offsets[]) {
  const int id_qpoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_qpoint >= query_points->num_points) return;

  // TODO - Idea to improve this kernel: Always only compare two local knns and do that in parallel and iteratively until only one knn array is left which is then the global knn

  // Set all elements in block_offsets to zero
  for (int j = 0; j < blockDimX_local_knn; ++j) {
    block_offsets[id_qpoint * blockDimX_local_knn + j] = 0;
  }

  // Find the global k nearest neighbors of from all the local k nearest neighbors
  // local_knns stores all local k nearest neighbors sequentially, e.g. from index 0 to k-1 you have
  // the local k nearest neighbors of thread block 0, from index k to 2k-1 you have the local k nearest
  // neighbors of thread block 1, and so on. The local k nearest neighbors are sorted in ascending order,
  // e.g. index 0 stores the nearest neighbor of thread block 0.
  // Hence, to determine the global nearest neighbor we only have to check the first element of all
  // k nearest neighbors. Then, if we want to determine the (global) second nearest neighbor we have to do
  // the same but we have to consider that we have already used the first element of one of the local k
  // nearest neighbors. This is done by adding an individual offset value to every local k nearest
  // neighbors subarray. These offset values are stored in 'block_offsets'.
  for (int i = 0; i < k; ++i) {
    float min_distance = FLT_MAX; // initialize to max value
    int min_dist_block_id;
    for (int j = 0; j < blockDimX_local_knn; ++j) {
      int block_offset = block_offsets[id_qpoint * blockDimX_local_knn + j];
      if (block_offset >= 0) { // enforce that we can only use TPB_LOCAL_KNN elements from every local knn subarray
        float min_dist_block = local_knns_distances[k * blockDimX_local_knn * id_qpoint + k * j + block_offset]; // minimum unused distance of local knn subarray
        if (min_dist_block < min_distance) {
          min_distance = min_dist_block;
          min_dist_block_id = j;
        }
      }
    }

    int min_dist_block_offset = block_offsets[id_qpoint * blockDimX_local_knn + min_dist_block_id];
    int min_dist_index = k * blockDimX_local_knn * id_qpoint + min_dist_block_id * k + min_dist_block_offset;
    query_points->neighbor_idx[k * id_qpoint + i] = local_knns_idx[min_dist_index]; // = id_datapoint
    query_points->neighbor_distances[k * id_qpoint + i] = local_knns_distances[min_dist_index];
    if (min_dist_block_offset < (TPB_LOCAL_KNN_X - 1)) {
      block_offsets[id_qpoint * blockDimX_local_knn + min_dist_block_id] += 1;
    } else {
      // Handle the case k > TPB_LOCAL_KNN_X -> Then we need this to enforce that we can only
      // use TPB_LOCAL_KNN_X elements from every local knn subarray
      block_offsets[id_qpoint * blockDimX_local_knn + min_dist_block_id] = -1;
    }
    
    #if DEBUG
    int id_datapoint = query_points->neighbor_idx[k * id_qpoint + i];
    printf("[DEBUG] calculate_global_knn | q_points->n_distances[%d] = %.4f \t id_datapoint = %d \t category = %d \t id_qpoint = %d\n", k * id_qpoint + i, query_points->neighbor_distances[k * id_qpoint + i], id_datapoint, dataset->categories[id_datapoint], id_qpoint);
    #endif
  }
}

/**
 * Determine the classes of multiple comparison points given their global k nearest neighbors
 * (global knns) by determining the majority class of the global knns. Each thread handles one
 * comparison point.
 *
 * @param k Number of neighbor to consider to determine the k nearest neighbors.
 * @param num_cpoints Number of comparison points.
 * @param global_knn Array containing the global knns.
 * @param cpoint_classes Array in which the results, i.e. the classes of all comparison points,
 * are stored. Must be allocated before calling the function and must have size 'num_cpoints'.
 * Index 0 stores the class of the first query point, index 1 stores the class of the second
 * query point, and so on.
 */
__global__ void determine_majority_classes_parallel(const int k, Dataset const* dataset, Query_Points* query_points) {
  const int id_qpoint = blockIdx.x * blockDim.x + threadIdx.x;
  if (id_qpoint >= query_points->num_points) return;

  extern __shared__ int s_neighbour_categories[]; // size: k * blockDim.x * 4 byte (TODO: be careful because max. size of shared memory is 48KB)

  #if TIMER
  clock_t start, end;
  start = clock();
  double time_used;
  #endif

  for (int i = 0; i < k; ++i) {
    int point_id = query_points->neighbor_idx[id_qpoint * k + i];
    s_neighbour_categories[id_qpoint * k + i] = dataset->categories[point_id];
  }

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] determine_majority_classes_parallel - fill shared memory %f \n", time_used);

  start = clock();
  #endif

  query_points->qpoint_categories[id_qpoint] = most_frequent(s_neighbour_categories + k * id_qpoint, k);

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] determine_majority_classes_parallel -  %f \n", time_used);
  #endif

  #if DEBUG
  printf("[DEBUG] Predicted category = %d \t id_qpoint = %d\n", query_points->qpoint_categories[id_qpoint], id_qpoint);
  #endif
}

/**
 * Determine the classes of multiple comparison points given their global k nearest neighbors
 * (global knns) by determining the majority class of the global knns.
 *
 * @param k Number of neighbor to consider to determine the k nearest neighbors.
 * @param num_cpoints Number of comparison points.
 * @param global_knn Array containing the global knns.
 * @param cpoint_classes Array in which the results, i.e. the classes of all comparison points,
 * are stored. Must be allocated before calling the function and must have size 'num_cpoints'.
 * Index 0 stores the class of the first query point, index 1 stores the class of the second
 * query point, and so on.
 */
__host__ void determine_majority_classes(const int k, Dataset const* dataset, Query_Points* query_points) {
  for (int i = 0; i < query_points->num_points; ++i) {
    // Store the categories of the global k nearest neighbors in an array
    int neighbour_categories[k];
    for (int c = 0; c < k; c++) {
      int point_id = query_points->neighbor_idx[i * k + c];
      neighbour_categories[c] = dataset->categories[point_id];
    }
    query_points->qpoint_categories[i] = most_frequent(neighbour_categories, k);
    #if DEBUG
    printf("[DEBUG] Predicted category = %d \t id_qpoint = %d\n", query_points->qpoint_categories[i], i);
    #endif
  }
}

// Passing by reference is less safe, but as a result of the performance increase it is justified
__host__ __device__ void print_point(float const* point, const int category, const int num_dimensions) {
  printf("(");
  int i = 0;
  do {
    if (i > 0) {
      printf(", ");
    }
    printf("%.4f", point[i]);
    i++;
  } while(i < num_dimensions);
  printf(") %d\n", category);
}

// Large dataset shouldn't be copied to support large datasets
__host__ __device__ void print_dataset(Dataset* dataset) {
  printf("Dataset\nDimensionality: %d\nNumber of Points: %d\n", dataset->num_dimensions, dataset->num_points);
  for (int i = 0; i < dataset->num_points; ++i) {
    print_point(dataset->get_point(i), dataset->categories[i], dataset->num_dimensions);
  }
}

void print_classes(Classifier_List classes) {
  for (int i = 0; i < classes.num_categories; ++i) {
    printf("Categories: %s\n", classes.categories[i].str);
  }
}

/**
 * Print the dataset stored in GPU memory from within a global function.
 *
 * @param dataset Dataset used to determine the classes of the query points.
 */
__global__ void print_dataset_parallel(Dataset* dataset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) return;

  print_dataset(dataset);
}

/**
 * Doing a k nearest neighbour search using the GPU.
 *
 * The knn search is parallelized both over query points and datasets points.
 *
 * @param k Number of neighbor to consider to determine the k nearest neighbors.
 * @param cpoints Array containing the query points (= comparison points).
 * @param num_cpoints Number of comparison points.
 * @param dataset Dataset used to determine the classes of the query points.
 * @return Array containing the determined classes of the query points. Has to be freed after its usage.
 */
int* knn_search_parallel(const int k, Query_Points* query_points, Dataset const* dataset) {
  // Warn if k is even
  if (k % 2 == 0) {
    printf("[WARN] Warning: %d is even. Tie cases have undefined behaviour\n", k);
  }

  #if DEBUG
  printf("[DEBUG] k: %d\n", k);
  #endif

  #if DEBUG
  for (int i = 0; i < query_points->num_points; ++i) {
    printf("[DEBUG] Query point %d: ", i);
    print_point(query_points->get_point(i), -1, query_points->num_dimensions);
  }
  #endif

  // Declare pointers pointing to GPU memory
  Query_Points* query_points_device;
  Dataset* dataset_device;
  float* local_knns_distances_device;
  int* local_knns_idx_device;
  int* block_offsets;

  // GPU pointers required as helper variables for copying the data structures
  float* dataset_points_device;
  int* dataset_categories_device;

  float* query_points_points_device;
  int* query_points_neighbor_idx_device;
  float* query_points_neighbor_distances_device;
  int* query_points_qpoint_categories_device;

  // Allocate GPU memory
  // 1) Dataset
  cudaMalloc(&dataset_device, sizeof(Dataset));
  cudaMalloc(&dataset_points_device, dataset->num_points * dataset->num_dimensions * sizeof(float));
  cudaMalloc(&dataset_categories_device, dataset->num_points * sizeof(int));

  // 2) Query points
  cudaMalloc(&query_points_device, sizeof(Query_Points));
  cudaMalloc(&query_points_points_device, query_points->num_points * query_points->num_dimensions * sizeof(float));
  cudaMalloc(&query_points_neighbor_idx_device, query_points->num_points * k * sizeof(int));
  cudaMalloc(&query_points_neighbor_distances_device, query_points->num_points * k * sizeof(float));
  cudaMalloc(&query_points_qpoint_categories_device, query_points->num_points * sizeof(int));

  // Copy memory to GPU
  // 1) Dataset
  cudaMemcpy(dataset_device, dataset, sizeof(Dataset), cudaMemcpyHostToDevice);
  cudaMemcpy(dataset_points_device, dataset->points, dataset->num_points * dataset->num_dimensions * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dataset_device->points), &dataset_points_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(dataset_categories_device, dataset->categories, dataset->num_points * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(&(dataset_device->categories), &dataset_categories_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct

  // 2) Query points
  cudaMemcpy(query_points_device, query_points, sizeof(Query_Points), cudaMemcpyHostToDevice);
  cudaMemcpy(query_points_points_device, query_points->points, query_points->num_points * query_points->num_dimensions * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(query_points_device->points), &query_points_points_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->neighbor_idx), &query_points_neighbor_idx_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->neighbor_distances), &query_points_neighbor_distances_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(&(query_points_device->qpoint_categories), &query_points_qpoint_categories_device, sizeof(int*), cudaMemcpyHostToDevice); // bind pointer to struct

  #if DEBUG
  printf("[DEBUG] Dataset used for knn search:\n");
  print_dataset_parallel<<<1, 32>>>(dataset_device);
  #endif

  // Initialize the grid and block dimensions and calculate the local k nearest neighbors
  // Use 2D block and thread grid: x - dimension of the dataset points, y - dimension of the query points
  const int blockDimY_local_knn = (query_points->num_points + TPB_LOCAL_KNN_Y - 1) / TPB_LOCAL_KNN_Y;
  const int blockDimX_local_knn = (dataset->num_points + TPB_LOCAL_KNN_X - 1) / TPB_LOCAL_KNN_X;
  int smem_size = TPB_LOCAL_KNN_X * TPB_LOCAL_KNN_Y * sizeof(float);
  cudaMalloc(&local_knns_distances_device, query_points->num_points * k * blockDimX_local_knn * sizeof(float));
  cudaMalloc(&local_knns_idx_device, query_points->num_points * k * blockDimX_local_knn * sizeof(int));
  calculate_local_knns<<<dim3(blockDimX_local_knn, blockDimY_local_knn, 1), dim3(TPB_LOCAL_KNN_X, TPB_LOCAL_KNN_Y, 1), smem_size>>>(k, query_points_device, dataset_device, local_knns_distances_device, local_knns_idx_device);
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions and calculate the global k nearest neighbors
  cudaMalloc(&block_offsets, query_points->num_points * blockDimX_local_knn * sizeof(int));
  const int num_blocks_global_knn = (query_points->num_points + TPB_GLOBAL_KNN - 1) / TPB_GLOBAL_KNN;
  calculate_global_knn<<<num_blocks_global_knn, TPB_GLOBAL_KNN>>>(k, dataset_device, query_points_device, local_knns_distances_device, local_knns_idx_device, blockDimX_local_knn, block_offsets);
  cudaFree(block_offsets);
  cudaDeviceSynchronize();

  #if MAJORITY_CLASS_PARALLEL
  // Determine the category of the query points by determining the majority class of the global k nearest neighbors (in parallel)
  smem_size = k * TPB_GLOBAL_KNN * sizeof(int);
  nvtxRangePush("knn_search_parallel - determine_majority_classes_parallel (parallel)")
  determine_majority_classes_parallel<<<num_blocks_global_knn, TPB_GLOBAL_KNN, smem_size>>>(k, dataset_device, query_points_device);
  nvtxRangePop();
  // Copy the result from GPU memory to the CPU memory
  cudaMemcpy(query_points->neighbor_idx, query_points_neighbor_idx_device, k * query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(query_points->qpoint_categories, query_points_qpoint_categories_device, query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  #else
  // Copy the result from GPU memory to the CPU memory
  cudaMemcpy(query_points->neighbor_idx, query_points_neighbor_idx_device, k * query_points->num_points * sizeof(int), cudaMemcpyDeviceToHost);
  // Determine the category of the query points by determining the majority class of the global k nearest neighbors (sequentially)
  nvtxRangePush("knn_search_parallel - determine_majority_classes (sequential)")
  determine_majority_classes(k, dataset, query_points);
  nvtxRangePop();
  #endif

  // Free the GPU memory
  cudaFree(query_points_device);
  cudaFree(query_points_points_device);
  cudaFree(query_points_neighbor_idx_device);
  cudaFree(query_points_neighbor_distances_device);
  cudaFree(query_points_qpoint_categories_device);

  cudaFree(dataset_device);
  cudaFree(dataset_points_device);
  cudaFree(dataset_categories_device);

  cudaFree(local_knns_distances_device);
  cudaFree(local_knns_idx_device);

  return query_points->qpoint_categories;
}

//Function that takes in a classification integer, and returns a classification string
//Requires a map between the integers and the string in the form of a classification_map datatype
my_string classify(const Classifier_List category_map, const int category) {
  my_string class_string = category_map.categories[category];
  return class_string;
}

int count_fields(char const* buffer) {
  int count = 1;
  int pos = 0;
  char current;
  do {
    current = buffer[pos];
    // printf("%c", current);
    if (current == ',') {
      count++;
    }
    pos++;
  } while(current != '\n' && current != '\0');
  #if DEBUG
  printf("[DEBUG] Number of fields: %d\n", count);
  #endif
  return count;
}

int get_class_num(const my_string in_string, Classifier_List* class_list) {
  //Check to see if any of the strings are present in the classifier list
  //Could improve with a Levenshtein Distance calculation to account for human errors
  //Also, if i is zero, we won't even need to check ifit's in there, we know it's not
  #if DEBUG
  printf("[DEBUG] class_list->num_categories: %d\n", class_list->num_categories);
  #endif

  for (int i = 0; i < class_list->num_categories; i++) {
    if (strcmp(class_list->categories[i].str, in_string.str) == 0) {
      return i;
    }
  }
  //If it isn't present in the existing array, we need to add it in.
  //Increment the count of categories
  class_list->num_categories++;
  #if DEBUG
  printf("[DEBUG] Class list categories: %d\n", class_list->num_categories);
  #endif
  class_list->categories = (my_string*) realloc(class_list->categories, sizeof(my_string) * class_list->num_categories);
  class_list->categories[class_list->num_categories - 1] = in_string;
  return class_list->num_categories - 1;
}

//Function to read lines from CSV
//Takes a file name
my_string extract_field(my_string line, const int field) {
  my_string return_value;
  //Using https://support.microsoft.com/en-us/help/51327/info-strtok-c-function----documentation-supplement
  if (field > count_fields(line.str)) {
    strcpy(return_value.str, "\0");
    return return_value;
  }
  //Potentially unsafe
  char *token;

  token = strtok(line.str, " ,");
  //Call strtok "field" times
  //Return that value of the token
  for (int i = 1; i < field; i++) {
    #if DEBUG
    printf("[DEBUG] Token is: %s\n", token);
    #endif

    token = strtok(NULL, " ,");
    #if DEBUG
    printf("[DEBUG] Before copy in loop\n");
    #endif
  }
  strncpy(return_value.str, token, sizeof(return_value.str));

  return return_value;
}

int count_lines(const my_string filename) {
  FILE *file;
  if (access(filename.str, F_OK) == -1) {
    printf("[ERROR] Could not find file");
    return -1;
  }
  file = fopen(filename.str, "r");
  char buffer[1024];
  int count = 0;
  while(fgets(buffer, 1024, file)) {
    // True if line is non-empty (only \n means also non-empty)
    count++;
  }
  fclose(file);
  #if DEBUG
  printf("[DEBUG] Number of lines in input file: %d\n", count);
  #endif
  return count;
}

// function that takes in a line, and returns a point
void Dataset::parse_point(const int point_id, const my_string line, Classifier_List* class_list) {
  float* point = this->get_point(point_id);
  for (int i = 0; i < this->num_dimensions; ++i) {
    //Go through and pull out the first num fields, and construct a point out of them
    // pass the string into a function that just mocks out and returns 1
    //Since the extract_field function extracts with a base 1, rather than base of 0
    point[i] = atof(extract_field(line, i + 1).str);
  }

  // The data for the class is in the last column
  this->categories[point_id] = get_class_num(extract_field(line, num_dimensions + 1), class_list);
  #if DEBUG
  print_point(point, this->categories[point_id], this->num_dimensions);
  #endif
}

void Dataset::read_dataset_file(const my_string filename, Classifier_List* class_list) {
  // Read the number of lines in before opening the files
  this->num_points = count_lines(filename);
  
  // From that, it should return some struct
  FILE *file;
  if (access(filename.str, F_OK) == -1) {
    printf("[ERROR] Could not find file.");
  }
  file = fopen(filename.str, "r");

  // Struct should contain a 2d array with the lines, in each with data separated into array elements
  char *buffer;
  buffer = (char*) malloc(1024 * sizeof(char));
  (void)! fscanf(file, "%s\n", buffer);

  // Count the commas
  this->num_dimensions = count_fields(buffer) - 1;

  // Allocate memory
  this->points = (float*) malloc(this->num_points * this->num_dimensions * sizeof(float));
  this->categories = (int*) malloc(this->num_points * sizeof(int));

  my_string buffer_string;
  strcpy(buffer_string.str, buffer);

  int i = 0;
  //For each line, parse the point and add it to the dataset
  do {
    this->parse_point(i, buffer_string, class_list);

    ++i;
    //Don't do this on the last iteration of the loop
    if (!(i == this->num_points)) {
      (void)! fscanf(file, "%s\n", buffer);
      strcpy(buffer_string.str, buffer);
    }
  } while (i < this->num_points);

  // Now we can essentially read in the first "count" fields and cast to float
  // Read in the last field, IE count and add a class for the
  free(buffer);
}

Classifier_List new_classifier_list() {
  int num_categories = 0;
  my_string *categories;
  categories = (my_string*) malloc(sizeof(my_string));
  Classifier_List new_list = {categories, num_categories};
  return new_list;
}

// Takes k as a parameter and also a dataset
// Measure the accuracy of the knn given a dataset, using the remove one method
float evaluate_knn(const int k, Dataset* benchmark_dataset) {
  #if DEBUG
  printf("============================================\n");
  printf("[DEBUG] Complete dataset:\n");
  print_dataset(benchmark_dataset);
  #endif

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif

  float accuracy;
  Dataset comparison_dataset(benchmark_dataset->num_dimensions, benchmark_dataset->num_points - 1);
  comparison_dataset.points = (float*) malloc(comparison_dataset.num_points * comparison_dataset.num_dimensions * sizeof(float));
  comparison_dataset.categories = (int*) malloc(comparison_dataset.num_points * sizeof(int));

  int sum_correct = 0;
  // Make a copy of the dataset, except missing the i'th term.
  for (int i = 0; i < benchmark_dataset->num_points; i++) {
    
    //Loop through the dataset the number of times there are points
    #if DEBUG
    printf("============================================\n");
    printf("[DEBUG] i: %d\n", i);
    #endif
    for (int j = 0; j < comparison_dataset.num_points; j++) {
      //Don't copy the ith term
      //Index will point to the correct term
      int index;
      if (j >= i) {
        index = j + 1;
      } else {
        index = j;
      }
      #if DEBUG
      printf("[DEBUG] Index: %d\n", index);
      #endif

      // Copy point
      float* comparison_datapoint = comparison_dataset.get_point(j);
      float* benchmark_datapoint = benchmark_dataset->get_point(index);
      for (int dim = 0; dim < comparison_dataset.num_dimensions; ++dim) {
        comparison_datapoint[dim] = benchmark_datapoint[dim];
      }
      comparison_dataset.categories[j] = benchmark_dataset->categories[index];
    }

    // Create a query point out of that i'th term
    Query_Points query_point(false, benchmark_dataset->num_dimensions, 1, k);
    query_point.set_point(0, benchmark_dataset->get_point(i));
    #if DEBUG
    printf("[DEBUG] Gets to the knn search\n");
    #endif
    //if the classification matches the category, add it to a sum
    #if CUDA
    int* qpoint_categories = knn_search_parallel(k, &query_point, &comparison_dataset);
    if (qpoint_categories[0] == benchmark_dataset->categories[i]) {
      sum_correct++;
    }
    #else
    int* qpoint_categories = knn_search(k, &query_point, &comparison_dataset);
    if ( qpoint_categories[0] == benchmark_dataset->categories[i]) {
      sum_correct++;
    }
    #endif
    #if DEBUG
    printf("[DEBUG] Actual category: %d\n", benchmark_dataset->categories[i]);
    #endif

    free(query_point.points);
    free(query_point.neighbor_idx);
    free(query_point.neighbor_distances);
    free(query_point.qpoint_categories);
  }

  accuracy = (float) sum_correct / (float) benchmark_dataset->num_points;

  // Free CPU memory
  free(comparison_dataset.points);
  free(comparison_dataset.categories);

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] evaluate_knn - execution time for k = %d neigbors: %f sec\n", k, time_used);
  #endif

  return accuracy;
}

#ifndef NDEBUG
//Definitions required for the testrunner
GREATEST_MAIN_DEFS();
#endif

//This main function takes commandline arguments
int main (int argc, char **argv) {
  srand(SEED);
  //Wrapped in #ifndef so we can make a release version
  #ifndef NDEBUG
  //Setup required testing
    GREATEST_MAIN_BEGIN();

    //Runs tests from external file specified above
    RUN_SUITE(external_suite);

    //Show results of the testing
    GREATEST_MAIN_END();
  #endif

  Classifier_List class_list = new_classifier_list();
  my_string filename = read_string("Filename: ");
  Dataset generic_dataset(filename, &class_list);

  int k = read_integer("Please put the desired number of neighbours k for the search: ");

  #if !EVALUATE

  int num_query_points = 1;
  num_query_points = read_integer("How many query points do you want to enter?: ");
  bool query_points_manually = read_boolean("Do you want to enter the query points manually? (yes/no) If no, the query points will be chosen randomly: ");
  
  Query_Points query_points(false, generic_dataset.num_dimensions, num_query_points, k);

  // loop to create the number of required query points
  for(int i = 0; i < num_query_points; ++i){
    query_points.read_query_point_user(i, query_points_manually);
  }

  #if CUDA

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif

  
  nvtxRangePush("main - knn_search_parallel (parallel)");
  int* qpoint_categories = knn_search_parallel(k, &query_points, &generic_dataset);
  nvtxRangePop();

  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] main - knn_search_parallel - execution time for %d k neigbours and %d query points: %f \n", k, num_query_points, time_used);
  #endif

  #else

  #if TIMER
  clock_t start, end;
  double time_used;
  start = clock();
  #endif
  nvtxRangePush("main - knn_search (sequential)");
  int* qpoint_categories = knn_search(k, &query_points, &generic_dataset);
  nvtxRangePop();
  #if TIMER
  end = clock();
  time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("[TIMER] main - knn_search - execution time for %d k neigbours and %d query points: %f \n", k, num_query_points, time_used);
  #endif

  #endif

  #if DEBUG
  printf("[DEBUG] Category is: %d\n", category);
  #endif
  for(int j = 0; j < num_query_points; ++j){
    my_string class_string = classify(class_list, qpoint_categories[j]);
    printf("Query point ID %d classified as: %s\n", j, class_string.str);
  }

  free(query_points.points);
  free(query_points.neighbor_idx);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);
  

  #endif

  #if EVALUATE
  for (int k = 1; k < generic_dataset.num_points; k = k + 2) {
    printf("k: %d, accuracy: %.4f\n", k, evaluate_knn(k, &generic_dataset));
    #if DEBUG
    printf("++++++++++++++++++++++++++++++++++++++++++++\n\n");
    #endif
  }
  //for values of k up to the number of points that exist in the dataset
  #endif

  // Free CPU memory
  free(class_list.categories);
  free(generic_dataset.points);
  free(generic_dataset.categories);

  return 0;
>>>>>>> 09c7a9287694984fddbb27faf1b51f6e8306a8b1
}