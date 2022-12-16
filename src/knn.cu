#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

#include "greatest.h"
#include "terminal_user_input.h"

#define EVALUATE 1  // Choose between evaluation mode (1) and user mode (0)
#define CUDA 1      // Choose between parallel/CUDA mode (1) and sequential mode (0)
#define DEBUG 0     // Define the debug level. Outputs verbose output if enabled (1) and not if disabled (0)

#define TPB_LOCAL_KNN 32   // Threads per block for calculating the local k nearest neighbors
#define TPB_GLOBAL_KNN 32  // Threads per block for calculating the global k nearest neighbors and determining the class

//Define a testing suite that is external to reduce code in this file
SUITE_EXTERN(external_suite);

//Datatype allows classifications to be stored very efficiently
//Is an array of char *, which is a double char *
//In order to use this struct, you must first define an array of char* on the class
typedef struct {
  my_string *categories;
  int num_categories;
} Classifier_List;

//Datatype is euclidean point
typedef struct point {
  float *dimension;
  //category must be in the categories array
  int category;
} Point;

//Dataset holds all of the points
typedef struct dataset {
  //d - the dimensionality of the dataset
  int dimensionality;
  int num_points;
  Point* points;
} Dataset;

//Distance holds the distance from a point, to another point
typedef struct point_neighbour_relationship {
  float distance;
  Point *neighbour_pointer;
} Point_Neighbour_Relationship;

//Since a comparison point is a distinctly different entity to a data point
typedef struct comparision_point {
  float *dimension;
  Point_Neighbour_Relationship *neighbour;
} Comparison_Point;

//Distance
//Return: number with the distance, a float
//Inputs, euclidean point, x and euclidean point y
__host__ __device__ float point_distance(Comparison_Point x, Point y, int dimensions) {
  float dist = 0;

  float sum = 0;

  //for each element in each, get the squared difference
  for (int i = 0; i < dimensions; i++) {
    //sum this squared difference
    sum = sum + pow(x.dimension[i] - y.dimension[i], 2);
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

//Calculate the mode
int mode(int *values, int num_values) {
  //Sort the array
  int current_counter = 1;
  int max_count = 1;
  int max_index = 0;
  //Count the number of each number
  qsort(values, num_values, sizeof(int), compare_int);
  #if DEBUG
  printf("[DEBUG] Values[%d]: %d\n", 0, values[0]);
  #endif
  for (int i = 1; i < num_values; i++) {
    //if this is the same as the last
    if (values[i-1] == values[i]) {
      //increase the couter
      current_counter += 1;
    } else if (current_counter > max_count) {
      //if the counter is greater than the max counter
      //set the max counter to counter
      max_count = current_counter;

      //update the max_index
      max_index = i - 1;
      #if DEBUG
      printf("[DEBUG] Max index updated to %d\n", i - 1);
      #endif

      //set the counter to 0
      current_counter = 0;
    }
    //If it's the last one, and the loop doesn't go through again
    if (current_counter > max_count) {
      //if the counter is greater than the max counter
      //set the max counter to counter
      max_count = current_counter;

      //update the max_index
      max_index = i;
      #if DEBUG
      printf("[DEBUG] Max index updated to %d\n", i - 1);
      #endif
    }

    //Keep a reference to an instance of the highest counted number in the array
    #if DEBUG
    printf("[DEBUG] Values[%d]: %d\n", i, values[i]);
    #endif
  }

  return values[max_index];
}

// Find the most frequent element in a C array (https://www.geeksforgeeks.org/frequent-element-array/)
__host__ __device__ int most_frequent(int* arr, const int n) {
  int maxcount = 0;
  int element_having_max_freq;
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
int knn_search(int k, Comparison_Point compare, Dataset *datapoints) {
  //Warn if k is even
  if (k % 2 == 0) {
    printf("[WARN] Warning: %d is even. Tie cases have undefined behaviour\n", k);
  }

  #if DEBUG
  printf("[DEBUG] k: %d\n", k);
  #endif

  //create an array the length of k to put all of the compared points in
  compare.neighbour = (Point_Neighbour_Relationship*) malloc(k*sizeof(Point_Neighbour_Relationship));
  //For the first k points, just add whatever junk into the array. This way we can just update the largest.
  for (int i = 0; i < k; i++) {
    float distance = point_distance(compare, datapoints->points[i], datapoints->dimensionality);
    compare.neighbour[i].distance = distance;
    compare.neighbour[i].neighbour_pointer = datapoints->points+i;
  }

  //Get the euclidean distance to every neighbour,
  for (int i = k; i < datapoints->num_points; i++) {
    float distance = point_distance(compare, datapoints->points[i], datapoints->dimensionality);

    #if DEBUG
    printf("[DEBUG] Point distance: %.4f\n", distance);
    #endif

    //if the neighbour is closer than the last, or it's null pointer distance closest keep it in a distance array
    //loop through all of the values for k, and keep the value from the comparison list for the compare point which is update_index.
    float max = 0;
    int update_index = 0;
    for (int j = 0; j < k; j++) {
      if (compare.neighbour[j].distance > max) {
        max = compare.neighbour[j].distance;
        update_index = j;
      }
      #if DEBUG
      printf("[DEBUG] Distance[%d]: %.4f\n", j, compare.neighbour[j].distance);
      #endif
    }
    #if DEBUG
    printf("[DEBUG] update_index max distance identified to be: %d at distance: %.4f\n", update_index, compare.neighbour[update_index].distance);
    #endif

    //if the current point distance is less than the largest recorded distance, or if the distances haven't been set
    if (compare.neighbour[update_index].distance > distance) {
      //Update the distance at update_index
      #if DEBUG
      //printf("[DEBUG] Neighbour number: %d is either null or distance is shorter, updating pointer\n", i);
      printf("[DEBUG] Compare neighbour[%d] = %.4f\n", update_index, distance);
      #endif
      compare.neighbour[update_index].distance = distance;

      //compare.neighbour[i].neighbour_pointer = &datapoints->points[i];
      compare.neighbour[update_index].neighbour_pointer = datapoints->points+i;

      #if DEBUG
      printf("[DEBUG] category of new point: %d\n", datapoints->points[i].category);
      #endif
    }
    #if DEBUG
    printf("============================================\n");
    #endif
  }
  //Now find the most frequently occurring neighbour pointer type
  //first get all the neighbour pointer categories and put them into a neighbour list
  int neighbour_categories[k];

  for (int c = 0; c < k; c++) {
    neighbour_categories[c] = compare.neighbour[c].neighbour_pointer->category;

    #if DEBUG
    printf("[DEBUG] compare.neighbour[%d].distance: %.4f\n", c, compare.neighbour[c].distance);
    printf("[DEBUG] Category[%d]: %d\n", c, neighbour_categories[c]);
    #endif
  }

  //Free memory
  free(compare.neighbour);

  //Find the mode of the categories
  //Call fuction with array of int and the length of the array and return the result
  //int category = mode(neighbour_categories, k);
  int category = most_frequent(neighbour_categories, k); // TODO
  #if DEBUG
  printf("[DEBUG] Determined class: %d\n", category);
  #endif
  return category;
}

// Calculate the local k nearest neighbors of a query point (= compare) given the dataset.
// The result is stored in the array local_knns. The array local_knns must have the size (k * <number of thread blocks>).
// Local k nearest neighbors means that for every thread block you consider a subset of the dataset and then you calculate
// for every thread block the k nearest neighbors in this subset.
__global__ void calculate_local_knns(const int k, Comparison_Point* compare, Dataset* datapoints, Point_Neighbour_Relationship* local_knns) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid_datapoint = true;
  if (idx >= datapoints->num_points) {
    valid_datapoint = false;
  }

  extern __shared__ float s_distances[]; // smem_size = blockDim.x * sizeof(float)

  // Calculate distances (valid points have the actual distance, invalid points have the max. possible distance)
  float distance = FLT_MAX;
  if (valid_datapoint) {
    distance = point_distance(*compare, datapoints->points[idx], datapoints->dimensionality);
  }
  
  s_distances[threadIdx.x] = distance;
  __syncthreads();

  // Determine the local k nearest neighbors
  // If rank_distance=3, it means that the corresponding point is the 4th closest datapoint to the
  // query point of all datapoints considered in this thread block.
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
  printf("[DEBUG] Point ID (idx) = %d \t rank_distance = %d \t\t distance = %.4f\n", idx, rank_distance, distance);
  #endif

  // Check if the calculated distance in this thread belongs to the local k nearest neighbors and if yes,
  // add the corresponding point to the array storing the local k nearest neighbors
  if (rank_distance < k) {
    Point_Neighbour_Relationship* local_nearest_neighbor = &(local_knns[k * blockIdx.x + rank_distance]);
    local_nearest_neighbor->neighbour_pointer = &(datapoints->points[idx]);
    local_nearest_neighbor->distance = distance;
    #if DEBUG
    printf("[DEBUG] local_knns[%d].distance = %.4f \t Point ID (idx) = %d \t Category: %d\n", k * blockIdx.x + rank_distance, local_nearest_neighbor->distance, idx, local_nearest_neighbor->neighbour_pointer->category);
    #endif
  }
}

// Calculate the global k nearest neighbors of a query point given its local k nearest neighbors.
// The result is stored in the variable query_classes. The array block_offsets is required within the function
// and its size must be equivalent to the number of local k nearest neighbors (= num_local_knn) multplied with
// the number of query points (= num_query_points).
__global__ void calculate_global_knn(const int k, Point_Neighbour_Relationship* local_knns, Point_Neighbour_Relationship* global_knn,
                                     const int num_local_knn, const int num_query_points, int* block_offsets) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_query_points) return;

  // TODO - Idea to improve this kernel: Always only compare two local knns and do that in parallel and iteratively until
  // only one knn array is left which is then the global knn

  // Set all elements in block_offsets to zero
  for (int j = 0; j < num_local_knn; ++j) {
    block_offsets[idx * num_local_knn + j] = 0;
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
    for (int j = 0; j < num_local_knn; ++j) {
      int block_offset = block_offsets[idx * num_local_knn + j];
      if (block_offset >= 0) { // enforce that we can only use TPB_LOCAL_KNN elements from every local knn subarray
        float min_dist_block = local_knns[j * k + block_offset].distance; // minimum unused distance of local knn subarray
        if (min_dist_block < min_distance) {
          min_distance = min_dist_block;
          min_dist_block_id = j;
        }
      }
    }

    int min_dist_block_offset = block_offsets[idx * num_local_knn + min_dist_block_id];
    int min_dist_index = min_dist_block_id * k + min_dist_block_offset;
    global_knn[i] = local_knns[min_dist_index];
    if (min_dist_block_offset < (TPB_LOCAL_KNN - 1)) {
      block_offsets[idx * num_local_knn + min_dist_block_id] += 1;
    } else {
      // Handle the case k > TPB_LOCAL_KNN -> Then we need this to enforce that we can only
      // use TPB_LOCAL_KNN elements from every local knn subarray
      block_offsets[idx * num_local_knn + min_dist_block_id] = -1;
    }
    
    #if DEBUG
    printf("[DEBUG] global_knn[%d].distance = %.4f \t min_dist_index = %d \t category: %d\n", i, global_knn[i].distance, min_dist_index, global_knn[i].neighbour_pointer->category);
    #endif
  }
}

// Determine the class of a query point given its global k nearest neighbors. The result is stored in the variable query_classes.
__global__ void determine_class(const int k, Point_Neighbour_Relationship* global_knn, const int num_query_points, int* query_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_query_points) return;

  extern __shared__ int s_neighbour_categories[]; // size: k * blockDim.x * 4 byte (TODO: be careful because max. size of shared memory is 48KB)

  for (int i = 0; i < k; ++i) {
    s_neighbour_categories[idx * k + i] = global_knn[i].neighbour_pointer->category;
  }

  query_classes[idx] = most_frequent(s_neighbour_categories + idx * k, k);
  #if DEBUG
  printf("[DEBUG] Determined class: %d\n", query_classes[idx]);
  #endif
}

//Passing by reference is less safe, but as a result of the performance increase it is justified
__host__ __device__ void print_point(Point *point_arg, int dimensions) {
  printf("(");
  int i = 0;
  do {
    if (i > 0) {
      printf(", ");
    }
    printf("%.4f", point_arg->dimension[i]);
    i++;
  } while(i < dimensions);
  printf(") %d\n", point_arg->category);
}

//Large dataset shouldn't be copied to support large datasets
__host__ __device__ void print_dataset(Dataset *dataset_arg) {
  printf("Dataset\nDimensionality: %d\nNumber of Points: %d\n", dataset_arg->dimensionality, dataset_arg->num_points);
  for (int i = 0; i < dataset_arg->num_points; i++) {
    print_point(dataset_arg->points + i, dataset_arg->dimensionality);
  }
}

void print_classes(Classifier_List classes) {
  for (int i = 0; i < classes.num_categories; i++) {
    printf("Categories: %s\n", classes.categories[i].str);
  }
}

// Print the dataset stored in GPU memory from within a global function
__global__ void print_dataset_parallel(Dataset *dataset_arg) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) return;

  print_dataset(dataset_arg);
}

// Doing a k nearest neighbour search using the GPU
int knn_search_parallel(int k, Comparison_Point compare, Dataset *datapoints) {
  // Warn if k is even
  if (k % 2 == 0) {
    printf("[WARN] Warning: %d is even. Tie cases have undefined behaviour\n", k);
  }

  #if DEBUG
  printf("[DEBUG] k: %d\n", k);
  #endif

  // TODO: Change code so that multiple queries can be handled at once
  const int num_query_points = 1;

  // Declare GPU pointers
  int* query_classes_host; // storing the result (determined classes)
  Comparison_Point* compare_device;
  Dataset* datapoints_device;
  Point_Neighbour_Relationship* local_knns_device;
  Point_Neighbour_Relationship* global_knn_device;

  // GPU pointers required for deep copying the compare and datapoints variables
  float* compare_point_device;
  Point* datapoints_points_device;
  float* point_dimensions_device[datapoints->num_points]; // array of pointers for storing the dimensions (x-, y-, ... values) of the points in the dataset

  // GPU pointers required as helper variables
  int* query_classes_device;
  int* block_offsets;

  // Allocate CPU memory
  query_classes_host = (int*) malloc(num_query_points * sizeof(int));

  // Allocate GPU memory
  cudaMalloc(&compare_device, sizeof(Comparison_Point));
  cudaMalloc(&compare_point_device, datapoints->dimensionality * sizeof(float));
  cudaMalloc(&datapoints_device, sizeof(Dataset));
  cudaMalloc(&datapoints_points_device, datapoints->num_points * sizeof(Point));
  for (int i = 0; i < datapoints->num_points; ++i) {
    cudaMalloc(&point_dimensions_device[i], datapoints->dimensionality * sizeof(float));
  }
  cudaMalloc(&global_knn_device, k * sizeof(Point_Neighbour_Relationship));
  cudaMalloc(&query_classes_device, num_query_points * sizeof(int));

  // Copy memory to the GPU
  cudaMemcpy(compare_device, &compare, sizeof(Comparison_Point), cudaMemcpyHostToDevice);
  cudaMemcpy(compare_point_device, compare.dimension, datapoints->dimensionality * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(&(compare_device->dimension), &compare_point_device, sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  cudaMemcpy(datapoints_device, datapoints, sizeof(Dataset), cudaMemcpyHostToDevice);
  cudaMemcpy(datapoints_points_device, datapoints->points, datapoints->num_points * sizeof(Point), cudaMemcpyHostToDevice);
  cudaMemcpy(&(datapoints_device->points), &datapoints_points_device, sizeof(Point*), cudaMemcpyHostToDevice); // bind pointer to struct
  for (int i = 0; i < datapoints->num_points; ++i) {
    cudaMemcpy(point_dimensions_device[i], datapoints->points[i].dimension, datapoints->dimensionality * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(datapoints_points_device[i].dimension), &point_dimensions_device[i], sizeof(float*), cudaMemcpyHostToDevice); // bind pointer to struct
  }

  #if DEBUG
  printf("[DEBUG] Dataset used for knn search:\n");
  print_dataset_parallel<<<1, 32>>>(datapoints_device);
  #endif

  // Initialize the grid and block dimensions and calculate the local k nearest neighbors (k nearest neighbors for every thread block)
  const int num_blocks_local_knn = (datapoints->num_points + TPB_LOCAL_KNN - 1) / TPB_LOCAL_KNN;
  int smem_size = TPB_LOCAL_KNN * sizeof(float);
  cudaMalloc(&local_knns_device, k * num_blocks_local_knn * sizeof(Point_Neighbour_Relationship));
  calculate_local_knns<<<num_blocks_local_knn, TPB_LOCAL_KNN, smem_size>>>(k, compare_device, datapoints_device, local_knns_device);
  cudaDeviceSynchronize();

  // Initialize the grid and block dimensions and calculate the global k nearest neighbors
  cudaMalloc(&block_offsets, num_query_points * num_blocks_local_knn * sizeof(int));
  const int num_blocks_global_knn = (num_query_points + TPB_GLOBAL_KNN - 1) / TPB_GLOBAL_KNN;
  calculate_global_knn<<<num_blocks_global_knn, TPB_GLOBAL_KNN>>>(k, local_knns_device, global_knn_device, num_blocks_local_knn, num_query_points, block_offsets);
  cudaDeviceSynchronize();

  // Determine the category of the query sample by determining the majority class of the k nearest neighbors
  smem_size = k * TPB_GLOBAL_KNN * sizeof(int);
  determine_class<<<num_blocks_global_knn, TPB_GLOBAL_KNN, smem_size>>>(k, global_knn_device, num_query_points, query_classes_device);

  // Copy the GPU memory back to the CPU
  cudaMemcpy(query_classes_host, query_classes_device, num_query_points * sizeof(int), cudaMemcpyDeviceToHost);

  // Free the GPU memory
  cudaFree(compare_device);
  cudaFree(compare_point_device);
  cudaFree(datapoints_device);
  cudaFree(datapoints_points_device);
  cudaFree(local_knns_device);
  cudaFree(global_knn_device);
  cudaFree(query_classes_device);

  return query_classes_host[0]; // TODO: query_classes_host needs to be freed
}

//Function that takes in a classification integer, and returns a classification string
//Requires a map between the integers and the string in the form of a classification_map datatype
my_string classify(Classifier_List category_map, int category) {
  my_string class_string = category_map.categories[category];
  return class_string;
}

Comparison_Point read_comparison_point_user(int num_dimensions) {
  Comparison_Point user_point;
  user_point.dimension = (float*) malloc(num_dimensions*sizeof(float));
  for (int i = 0; i < num_dimensions; i++) {
    printf("%dth dimension: ", i);
    user_point.dimension[i] = read_float("");
  }
  return user_point;
}

int count_fields(char *buffer) {
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

int get_class_num(my_string in_string, Classifier_List *class_list) {
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
my_string extract_field(my_string line, int field) {
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

int count_lines(my_string filename) {
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

Dataset new_dataset() {
  Point *points = {NULL};
  Dataset new_dataset = {0, 0, points};
  return new_dataset;
}

//function that takes in a line, and returns a point
//parse point
//TODO ADD UNIT TESTS!!!
Point parse_point(my_string line, int num_dimensions, Classifier_List *class_list) {
  float *dimensions = (float*) malloc(num_dimensions*sizeof(float));
  for (int i = 0; i < num_dimensions; i++) {
    //Go through and pull out the first num fields, and construct a point out of them
    // pass the string into a function that just mocks out and returns 1
    //Since the extract_field function extracts with a base 1, rather than base of 0
    dimensions[i] = atof(extract_field(line, i + 1).str);
  }

  Point curr_point;
  curr_point.dimension = dimensions;

  //Since the data for the class is one after the
  curr_point.category = get_class_num(extract_field(line, num_dimensions + 1), class_list);
  #if DEBUG
  print_point(&curr_point, num_dimensions);
  #endif
  return curr_point;
}

Dataset read_dataset_file(my_string filename, Classifier_List *class_list) {
  // Read the number of lines in before opening the files
  int num_lines = count_lines(filename);

  //From that, it should return some struct
  FILE *file;
  if (access(filename.str, F_OK) == -1) {
    printf("[ERROR] Could not find file.");
  }
  file = fopen(filename.str, "r");

  //Struct should contain a 2d array with the lines, in each with data separated into array elements
  char *buffer;
  buffer = (char*) malloc(sizeof(char) * 1024);
  fscanf(file, "%s\n", buffer);

  //Count the commas
  int num_dimensions = count_fields(buffer) - 1;

  //create a Dataset which can hold the rest of the data
  //dimensionality is the number of fields -1
  Point *points = (Point*) malloc(num_lines*sizeof(Point));
  Dataset return_dataset = {num_dimensions, num_lines, points};

  my_string buffer_string;
  strcpy(buffer_string.str, buffer);

  int i = 0;
  //For each line, parse the point and add it to the dataset
  do {
    points[i] = parse_point(buffer_string, num_dimensions, class_list);

    i++;
    //Don't do this on the last iteration of the loop
    if (!(i == num_lines)) {
      fscanf(file, "%s\n", buffer);
      strcpy(buffer_string.str, buffer);
    }
  } while (i < num_lines);

  // Now we can essentially read in the first "count" fields and cast to float
  // Read in the last field, IE count and add a class for the
  free(buffer);
  return return_dataset;
}

Classifier_List new_classifier_list() {
  int num_categories = 0;
  my_string *categories;
  categories = (my_string*) malloc(sizeof(my_string));
  Classifier_List new_list = {categories, num_categories};
  return new_list;
}

//Takes k as a parameter and also a dataset
//Measure the accuracy of the knn given a dataset, using the remove one method
float evaluate_knn(int k, Dataset *benchmark_dataset) {
  #if DEBUG
  printf("============================================\n");
  printf("[DEBUG] Complete dataset:\n");
  print_dataset(benchmark_dataset);
  #endif

  float accuracy;
  Dataset comparison_dataset = new_dataset();
  comparison_dataset.dimensionality = benchmark_dataset->dimensionality;
  comparison_dataset.num_points = benchmark_dataset->num_points - 1;

  comparison_dataset.points = (Point*) malloc(comparison_dataset.num_points*sizeof(Point));

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
      comparison_dataset.points[j] = benchmark_dataset->points[index];
    }
    //Create a comparison point out of that i'th term
    Comparison_Point compare = {benchmark_dataset->points[i].dimension, NULL};
    #if DEBUG
    printf("[DEBUG] Gets to the knn search\n");
    #endif
    //if the classification matches the category, add it to a sum
    #if CUDA
    if (knn_search_parallel(k, compare, &comparison_dataset) == benchmark_dataset->points[i].category) {
      sum_correct++;
    }
    #else
    if (knn_search(k, compare, &comparison_dataset) == benchmark_dataset->points[i].category) {
      sum_correct++;
    }
    #endif
    #if DEBUG
    printf("[DEBUG] Actual class: %d\n", benchmark_dataset->points[i].category);
    #endif
  }

  accuracy = (float) sum_correct / (float) benchmark_dataset->num_points;

  //Free CPU memory
  free(comparison_dataset.points);

  return accuracy;
}

#ifndef NDEBUG
//Definitions required for the testrunner
GREATEST_MAIN_DEFS();
#endif

//This main function takes commandline arguments
int main (int argc, char **argv) {
  // TODO: Remove this
  //printf("#################################################\n");
  //printf("{0, 0, 2, 1, 1, 1, 2}\n");
  //int test_array[] = {0, 0, 2, 1, 1, 1, 2};
  //int len = 7;
  //int test_mode = mode(test_array, len);
  //for (int i = 0; i < len; ++i) {
  //  printf("%d, ", test_array[i]);
  //}
  //printf("\ntest_mode = %d\n", test_mode);
  //int test_most_frequent = most_frequent(test_array, len);
  //printf("test_most_frequent = %d\n", test_most_frequent);
  //printf("#################################################\n\n");

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

  //This is in user mode:

  Dataset generic_dataset = read_dataset_file(filename, &class_list);

  #if !EVALUATE
  bool another_point = true;
  do {
    Comparison_Point compare = read_comparison_point_user(generic_dataset.dimensionality);
    int k = read_integer("k: ");
    #if CUDA
    int category = knn_search_parallel(k, compare, &generic_dataset);
    #else
    int category = knn_search(k, compare, &generic_dataset);
    #endif
    free(compare.dimension);
    compare.dimension = NULL;

    #if DEBUG
    printf("[DEBUG] Category is: %d\n", category);
    #endif

    my_string class_string = classify(class_list, category);
    printf("Point classified as: %s\n", class_string.str);
    another_point = read_boolean("Classify another point? ");
  } while(another_point);
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

  //Free CPU memory
  for (int i = 0; i < generic_dataset.num_points; ++i) {
    free(generic_dataset.points[i].dimension);
  }
  free(generic_dataset.points);
  free(class_list.categories);

  return 0;
}
