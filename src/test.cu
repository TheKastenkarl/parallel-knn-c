<<<<<<< HEAD
#include <math.h>
#include "knn.cu"
#include "greatest.h"
#include "terminal_user_input.h"

//Defining tolerances for tests
#define FLOAT_TOLERANCE 0.01

TEST most_frequent_3_inputs(void) {
  //Setup array of integers
  int inputs[3] = {3, 3, 7};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(3, most_frequent(inputs, 3));

  PASS();
}

TEST most_frequent_with_zero (void) {
  //Setup array of integers
  int inputs[3] = {0, 1, 1};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(1, most_frequent(inputs, 3));

  PASS();
}

TEST most_frequent_7_inputs(void) {
  //Setup array of integers
  int inputs[7] = {1, 2, 3, 1, 7, 8, 1};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(1, most_frequent(inputs, 7));

  PASS();
}

//Test bimodal

//Compare two integers that are equal
TEST compare_ints(void) {
  int n1 = 1;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 0);
  PASS();
}

//Compare two integers that are equal
TEST compare_greater_int(void) {
  int n1 = 2;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 1);
  PASS();
}

//Compare two integers that are equal
TEST compare_very_different_int_negative (void) {
  int n1 = 1;
  int n2 = 4;
  ASSERT_EQ(compare_int(&n1, &n2), -1);
  PASS();
}

TEST compare_very_different_int_positive (void) {
  int n1 = 4;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 1);
  PASS();
}

/* A test runs various assertions, then calls PASS(), FAIL(), or SKIP(). */
TEST distance_3_dimensions(void) {
  float array1[3] = {2.0, 2.0, 2.0};
  float array2[3] = {5.0, 5.0, 5.0};

  ASSERT_IN_RANGE(5.1962, point_distance(array1, array2, 3), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_10_dimensions(void) {
  float array1[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float array2[10] = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};

  ASSERT_IN_RANGE(28.4605, point_distance(array1, array2, 10), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_1_dimension(void) {
  float array1[1] = {3.0};
  float array2[1] = {6.0};

  ASSERT_IN_RANGE(3.0, point_distance(array1, array2, 1), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_1_dimension_fraction(void) {
  float array1[1] = {3.0};
  float array2[1] = {3.5};

  ASSERT_IN_RANGE(0.5, point_distance(array1, array2, 1), FLOAT_TOLERANCE);
  PASS();
}

//How do I initialise arrays with the {} curly braces syntax?

//Test the creation of an array (neighbours) with distances to every single point,
//taking one point and a dataset
//We need a distance associated with a point

// Test k NN search
// Ensure that the returned array contains the correct integers

// Test the nearest 1 neighbour can be found
TEST find_1_nearest_neighbour(void) {
  // Setup
  int k = 1;

  // Create dataset with a single 1D point
  int category = 0;
  float point[] = {5};
  Dataset single_point_dataset(1, 1);
  single_point_dataset.points = (float*) malloc(single_point_dataset.num_points * single_point_dataset.num_dimensions * sizeof(float));
  single_point_dataset.categories = (int*) malloc(single_point_dataset.num_points * sizeof(int));
  single_point_dataset.set_point(0, point);
  single_point_dataset.categories[0] = category;

  // Create one query point
  float query_point[] = {3};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(category, knn_search(k, &query_points, &single_point_dataset)[0]);

  free(single_point_dataset.points);
  free(single_point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_ids);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST find_1_nearest_neighbour_parallel(void) {
  // Setup
  int k = 1;

  // Create dataset with a single 1D point
  int category = 0;
  float point[] = {5};
  Dataset single_point_dataset(1, 1);
  single_point_dataset.points = (float*) malloc(single_point_dataset.num_points * single_point_dataset.num_dimensions * sizeof(float));
  single_point_dataset.categories = (int*) malloc(single_point_dataset.num_points * sizeof(int));
  single_point_dataset.set_point(0, point);
  single_point_dataset.categories[0] = category;

  // Create one query point
  float query_point[] = {3};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(category, knn_search_parallel(k, &query_points, &single_point_dataset)[0]);

  free(single_point_dataset.points);
  free(single_point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_ids);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

// One dimensional, 5 point dataset, find average of k=3 neighbours
// Test the code can handle updating 3 of the 5 without having to update a distance
TEST find_3_nearest_neighbour(void) {
  // Setup
  int k = 3;

  // Create dataset with 5x 1D points
  Dataset point_dataset(1, 5);
  point_dataset.points = (float*) malloc(point_dataset.num_points * point_dataset.num_dimensions * sizeof(float));
  point_dataset.categories = (int*) malloc(point_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  point_dataset.set_point(0, point0);
  point_dataset.categories[0] = 0;

  float point1[] = {6.0};
  point_dataset.set_point(1, point1);
  point_dataset.categories[1] = 1;

  float point2[] = {7.0};
  point_dataset.set_point(2, point2);
  point_dataset.categories[2] = 1;

  float point3[] = {0.0};
  point_dataset.set_point(3, point3);
  point_dataset.categories[3] = 0;

  float point4[] = {-1.0};
  point_dataset.set_point(4, point4);
  point_dataset.categories[4] = 0;

  // Create one query point
  float query_point[] = {6.5};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(1, knn_search(k, &query_points, &point_dataset)[0]);

  free(point_dataset.points);
  free(point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_ids);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST find_3_nearest_neighbour_parallel(void) {
  // Setup
  int k = 3;

  // Create dataset with 5x 1D points
  Dataset point_dataset(1, 5);
  point_dataset.points = (float*) malloc(point_dataset.num_points * point_dataset.num_dimensions * sizeof(float));
  point_dataset.categories = (int*) malloc(point_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  point_dataset.set_point(0, point0);
  point_dataset.categories[0] = 0;

  float point1[] = {6.0};
  point_dataset.set_point(1, point1);
  point_dataset.categories[1] = 1;

  float point2[] = {7.0};
  point_dataset.set_point(2, point2);
  point_dataset.categories[2] = 1;

  float point3[] = {0.0};
  point_dataset.set_point(3, point3);
  point_dataset.categories[3] = 0;

  float point4[] = {-1.0};
  point_dataset.set_point(4, point4);
  point_dataset.categories[4] = 0;

  // Create one query point
  float query_point[] = {6.5};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(1, knn_search_parallel(k, &query_points, &point_dataset)[0]);

  free(point_dataset.points);
  free(point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_ids);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST classify_int(void) {
  //The class integer to be selected
  int class_int = 0;

  //Using only the minimum 1 categories
  Classifier_List flower_map;
  flower_map.categories = (my_string*) malloc(sizeof(my_string));

  strcpy(flower_map.categories[0].str, "Iris");

  my_string class_string = classify(flower_map, class_int);

  ASSERT_STR_EQ("Iris", class_string.str);
  PASS();
  free(flower_map.categories);
}

TEST extract_field_1(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1, 1.2, 1.3, 1.4");

  ASSERT_STR_EQ("1.1", extract_field(test_line, 1).str);
  PASS();
}

TEST extract_field_4(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1, 1.2, 1.3, 1.4");

  ASSERT_STR_EQ("1.4", extract_field(test_line, 4).str);
  PASS();
}

TEST extract_field_different_formatting(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1,,,''1.2,three, 6");

  ASSERT_STR_EQ("three", extract_field(test_line, 3).str);
  PASS();
}

TEST extract_flower_field(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("Iris-setosa", extract_field(test_line, 5).str);
  PASS();
}

TEST field_2(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("3.5", extract_field(test_line, 2).str);
  PASS();
}

TEST out_of_bounds(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("\0", extract_field(test_line, 6).str);
  PASS();
}

TEST gets_class_int(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string strings[4] = {{"mycategory1"}, {"mycategory2"}, {"mycategory3"}, {"mycategory4"}};
  Classifier_List class_list = {strings, 4};
  ASSERT_EQ(0, get_class_num(class_list.categories[0], &class_list));

  PASS();
}

TEST initialise_category(void) {
  Classifier_List new_list = new_classifier_list();
  strcpy(new_list.categories[0].str, "Testing Category");

  ASSERT_EQ(0, new_list.num_categories);
  PASS();
}

TEST create_first_category(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string first_class = {"Test Category"};

  Classifier_List class_list = new_classifier_list();
  ASSERT_EQ(0, get_class_num(first_class, &class_list));
  ASSERT_STR_EQ(first_class.str, class_list.categories[0].str);

  PASS();
}

TEST create_new_category(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string first_class = {"Test Category"};
  my_string second_class = {"Class2"};

  Classifier_List class_list = new_classifier_list();
  ASSERT_EQ(0, get_class_num(first_class, &class_list));
  ASSERT_STR_EQ(first_class.str, class_list.categories[0].str);

  ASSERT_EQ(1, get_class_num(second_class, &class_list));
  ASSERT_STR_EQ(second_class.str, class_list.categories[1].str);

  #ifdef DEBUG
  print_classes(class_list);
  #endif

  PASS();
}

TEST knn_accuracy(void) {
  // Comments step through the expected classification of the knn
  // for each point removed and then consider the percentage correct for that k
  // In this case k=3

  // Create dataset with 5x 1D points
  Dataset test_dataset(1, 5);
  test_dataset.points = (float*) malloc(test_dataset.num_points * test_dataset.num_dimensions * sizeof(float));
  test_dataset.categories = (int*) malloc(test_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  test_dataset.set_point(0, point0);
  test_dataset.categories[0] = 0;
  //Classed 1
  //Incorrect

  float point1[] = {6.0};
  test_dataset.set_point(1, point1);
  test_dataset.categories[1] = 1;
  //Classed 0
  //Incorrect

  float point2[] = {7.0};
  test_dataset.set_point(2, point2);
  test_dataset.categories[2] = 1;
  //Classed 0
  //Incorrect

  float point3[] = {0.0};
  test_dataset.set_point(3, point3);
  test_dataset.categories[3] = 0;
  //Classed 0
  //Correct

  float point4[] = {-1.0};
  test_dataset.set_point(4, point4);
  test_dataset.categories[4] = 0;
  //Classed 0
  //Correct

  //Count is 2
  // 2/5=0.4

  evaluate_knn(3, &test_dataset);
  ASSERT_IN_RANGE(0.4, evaluate_knn(3, &test_dataset), FLOAT_TOLERANCE);

  free(test_dataset.points);
  free(test_dataset.categories);

  PASS();
}


//Test that the correct number is returned after a call to the string is passed to the classifier

/* Suites can group multiple tests with common setup. */
SUITE(external_suite) {
    RUN_TEST(distance_1_dimension);
    RUN_TEST(distance_1_dimension_fraction);
    RUN_TEST(distance_3_dimensions);
    RUN_TEST(distance_10_dimensions);

    RUN_TEST(classify_int);

    RUN_TEST(most_frequent_3_inputs);
    RUN_TEST(most_frequent_with_zero);
    RUN_TEST(most_frequent_7_inputs);

    RUN_TEST(compare_ints);
    RUN_TEST(compare_greater_int);
    RUN_TEST(compare_very_different_int_positive);
    RUN_TEST(compare_very_different_int_negative);

    RUN_TEST(find_1_nearest_neighbour);
    RUN_TEST(find_1_nearest_neighbour_parallel);
    RUN_TEST(find_3_nearest_neighbour);
    RUN_TEST(find_3_nearest_neighbour_parallel);

    RUN_TEST(extract_field_1);
    RUN_TEST(extract_field_4);
    RUN_TEST(extract_field_different_formatting);
    RUN_TEST(extract_flower_field);
    RUN_TEST(field_2);
    RUN_TEST(out_of_bounds);

    //Testing the configuration manager
    RUN_TEST(gets_class_int);
    RUN_TEST(initialise_category);
    RUN_TEST(create_first_category);
    RUN_TEST(create_new_category);

    RUN_TEST(knn_accuracy);
}

/* Add definitions that need to be in the test runner's main file. */
=======
#include <math.h>
#include "knn.cu"
#include "greatest.h"
#include "terminal_user_input.h"

//Defining tolerances for tests
#define FLOAT_TOLERANCE 0.01

TEST most_frequent_3_inputs(void) {
  //Setup array of integers
  int inputs[3] = {3, 3, 7};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(3, most_frequent(inputs, 3));

  PASS();
}

TEST most_frequent_with_zero (void) {
  //Setup array of integers
  int inputs[3] = {0, 1, 1};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(1, most_frequent(inputs, 3));

  PASS();
}

TEST most_frequent_7_inputs(void) {
  //Setup array of integers
  int inputs[7] = {1, 2, 3, 1, 7, 8, 1};

  //Pass array of integers into function
  //Check if most frequent element is correct
  ASSERT_EQ(1, most_frequent(inputs, 7));

  PASS();
}

//Test bimodal

//Compare two integers that are equal
TEST compare_ints(void) {
  int n1 = 1;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 0);
  PASS();
}

//Compare two integers that are equal
TEST compare_greater_int(void) {
  int n1 = 2;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 1);
  PASS();
}

//Compare two integers that are equal
TEST compare_very_different_int_negative (void) {
  int n1 = 1;
  int n2 = 4;
  ASSERT_EQ(compare_int(&n1, &n2), -1);
  PASS();
}

TEST compare_very_different_int_positive (void) {
  int n1 = 4;
  int n2 = 1;
  ASSERT_EQ(compare_int(&n1, &n2), 1);
  PASS();
}

/* A test runs various assertions, then calls PASS(), FAIL(), or SKIP(). */
TEST distance_3_dimensions(void) {
  float array1[3] = {2.0, 2.0, 2.0};
  float array2[3] = {5.0, 5.0, 5.0};

  ASSERT_IN_RANGE(5.1962, point_distance(array1, array2, 3), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_10_dimensions(void) {
  float array1[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float array2[10] = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};

  ASSERT_IN_RANGE(28.4605, point_distance(array1, array2, 10), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_1_dimension(void) {
  float array1[1] = {3.0};
  float array2[1] = {6.0};

  ASSERT_IN_RANGE(3.0, point_distance(array1, array2, 1), FLOAT_TOLERANCE);
  PASS();
}

TEST distance_1_dimension_fraction(void) {
  float array1[1] = {3.0};
  float array2[1] = {3.5};

  ASSERT_IN_RANGE(0.5, point_distance(array1, array2, 1), FLOAT_TOLERANCE);
  PASS();
}

//How do I initialise arrays with the {} curly braces syntax?

//Test the creation of an array (neighbours) with distances to every single point,
//taking one point and a dataset
//We need a distance associated with a point

// Test k NN search
// Ensure that the returned array contains the correct integers

// Test the nearest 1 neighbour can be found
TEST find_1_nearest_neighbour(void) {
  // Setup
  int k = 1;

  // Create dataset with a single 1D point
  int category = 0;
  float point[] = {5};
  Dataset single_point_dataset(1, 1);
  single_point_dataset.points = (float*) malloc(single_point_dataset.num_points * single_point_dataset.num_dimensions * sizeof(float));
  single_point_dataset.categories = (int*) malloc(single_point_dataset.num_points * sizeof(int));
  single_point_dataset.set_point(0, point);
  single_point_dataset.categories[0] = category;

  // Create one query point
  float query_point[] = {3};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(category, knn_search(k, &query_points, &single_point_dataset)[0]);

  free(single_point_dataset.points);
  free(single_point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_idx);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST find_1_nearest_neighbour_parallel(void) {
  // Setup
  int k = 1;

  // Create dataset with a single 1D point
  int category = 0;
  float point[] = {5};
  Dataset single_point_dataset(1, 1);
  single_point_dataset.points = (float*) malloc(single_point_dataset.num_points * single_point_dataset.num_dimensions * sizeof(float));
  single_point_dataset.categories = (int*) malloc(single_point_dataset.num_points * sizeof(int));
  single_point_dataset.set_point(0, point);
  single_point_dataset.categories[0] = category;

  // Create one query point
  float query_point[] = {3};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(category, knn_search_parallel(k, &query_points, &single_point_dataset)[0]);

  free(single_point_dataset.points);
  free(single_point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_idx);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

// One dimensional, 5 point dataset, find average of k=3 neighbours
// Test the code can handle updating 3 of the 5 without having to update a distance
TEST find_3_nearest_neighbour(void) {
  // Setup
  int k = 3;

  // Create dataset with 5x 1D points
  Dataset point_dataset(1, 5);
  point_dataset.points = (float*) malloc(point_dataset.num_points * point_dataset.num_dimensions * sizeof(float));
  point_dataset.categories = (int*) malloc(point_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  point_dataset.set_point(0, point0);
  point_dataset.categories[0] = 0;

  float point1[] = {6.0};
  point_dataset.set_point(1, point1);
  point_dataset.categories[1] = 1;

  float point2[] = {7.0};
  point_dataset.set_point(2, point2);
  point_dataset.categories[2] = 1;

  float point3[] = {0.0};
  point_dataset.set_point(3, point3);
  point_dataset.categories[3] = 0;

  float point4[] = {-1.0};
  point_dataset.set_point(4, point4);
  point_dataset.categories[4] = 0;

  // Create one query point
  float query_point[] = {6.5};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(1, knn_search(k, &query_points, &point_dataset)[0]);

  free(point_dataset.points);
  free(point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_idx);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST find_3_nearest_neighbour_parallel(void) {
  // Setup
  int k = 3;

  // Create dataset with 5x 1D points
  Dataset point_dataset(1, 5);
  point_dataset.points = (float*) malloc(point_dataset.num_points * point_dataset.num_dimensions * sizeof(float));
  point_dataset.categories = (int*) malloc(point_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  point_dataset.set_point(0, point0);
  point_dataset.categories[0] = 0;

  float point1[] = {6.0};
  point_dataset.set_point(1, point1);
  point_dataset.categories[1] = 1;

  float point2[] = {7.0};
  point_dataset.set_point(2, point2);
  point_dataset.categories[2] = 1;

  float point3[] = {0.0};
  point_dataset.set_point(3, point3);
  point_dataset.categories[3] = 0;

  float point4[] = {-1.0};
  point_dataset.set_point(4, point4);
  point_dataset.categories[4] = 0;

  // Create one query point
  float query_point[] = {6.5};
  Query_Points query_points(false, 1, 1, 1);
  query_points.set_point(0, query_point);

  // One point to compare to the rest
  ASSERT_EQ(1, knn_search_parallel(k, &query_points, &point_dataset)[0]);

  free(point_dataset.points);
  free(point_dataset.categories);
  free(query_points.points);
  free(query_points.neighbor_idx);
  free(query_points.neighbor_distances);
  free(query_points.qpoint_categories);

  PASS();
}

TEST classify_int(void) {
  //The class integer to be selected
  int class_int = 0;

  //Using only the minimum 1 categories
  Classifier_List flower_map;
  flower_map.categories = (my_string*) malloc(sizeof(my_string));

  strcpy(flower_map.categories[0].str, "Iris");

  my_string class_string = classify(flower_map, class_int);

  ASSERT_STR_EQ("Iris", class_string.str);
  PASS();
  free(flower_map.categories);
}

TEST extract_field_1(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1, 1.2, 1.3, 1.4");

  ASSERT_STR_EQ("1.1", extract_field(test_line, 1).str);
  PASS();
}

TEST extract_field_4(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1, 1.2, 1.3, 1.4");

  ASSERT_STR_EQ("1.4", extract_field(test_line, 4).str);
  PASS();
}

TEST extract_field_different_formatting(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "1.1,,,''1.2,three, 6");

  ASSERT_STR_EQ("three", extract_field(test_line, 3).str);
  PASS();
}

TEST extract_flower_field(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("Iris-setosa", extract_field(test_line, 5).str);
  PASS();
}

TEST field_2(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("3.5", extract_field(test_line, 2).str);
  PASS();
}

TEST out_of_bounds(void) {
  //From a string of "1.1, 1.2, 1.3, 1.4", extract field 1
  my_string test_line;
  strcpy(test_line.str, "5.1,3.5,1.4,0.2,Iris-setosa");

  ASSERT_STR_EQ("\0", extract_field(test_line, 6).str);
  PASS();
}

TEST gets_class_int(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string strings[4] = {{"mycategory1"}, {"mycategory2"}, {"mycategory3"}, {"mycategory4"}};
  Classifier_List class_list = {strings, 4};
  ASSERT_EQ(0, get_class_num(class_list.categories[0], &class_list));

  PASS();
}

TEST initialise_category(void) {
  Classifier_List new_list = new_classifier_list();
  strcpy(new_list.categories[0].str, "Testing Category");

  ASSERT_EQ(0, new_list.num_categories);
  PASS();
}

TEST create_first_category(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string first_class = {"Test Category"};

  Classifier_List class_list = new_classifier_list();
  ASSERT_EQ(0, get_class_num(first_class, &class_list));
  ASSERT_STR_EQ(first_class.str, class_list.categories[0].str);

  PASS();
}

TEST create_new_category(void) {
  //Pass in a string, with a class_list which contains it, see if the correct value is returned
  my_string first_class = {"Test Category"};
  my_string second_class = {"Class2"};

  Classifier_List class_list = new_classifier_list();
  ASSERT_EQ(0, get_class_num(first_class, &class_list));
  ASSERT_STR_EQ(first_class.str, class_list.categories[0].str);

  ASSERT_EQ(1, get_class_num(second_class, &class_list));
  ASSERT_STR_EQ(second_class.str, class_list.categories[1].str);

  #ifdef DEBUG
  print_classes(class_list);
  #endif

  PASS();
}

TEST knn_accuracy(void) {
  // Comments step through the expected classification of the knn
  // for each point removed and then consider the percentage correct for that k
  // In this case k=3

  // Create dataset with 5x 1D points
  Dataset test_dataset(1, 5);
  test_dataset.points = (float*) malloc(test_dataset.num_points * test_dataset.num_dimensions * sizeof(float));
  test_dataset.categories = (int*) malloc(test_dataset.num_points * sizeof(int));

  float point0[] = {5.0};
  test_dataset.set_point(0, point0);
  test_dataset.categories[0] = 0;
  //Classed 1
  //Incorrect

  float point1[] = {6.0};
  test_dataset.set_point(1, point1);
  test_dataset.categories[1] = 1;
  //Classed 0
  //Incorrect

  float point2[] = {7.0};
  test_dataset.set_point(2, point2);
  test_dataset.categories[2] = 1;
  //Classed 0
  //Incorrect

  float point3[] = {0.0};
  test_dataset.set_point(3, point3);
  test_dataset.categories[3] = 0;
  //Classed 0
  //Correct

  float point4[] = {-1.0};
  test_dataset.set_point(4, point4);
  test_dataset.categories[4] = 0;
  //Classed 0
  //Correct

  //Count is 2
  // 2/5=0.4

  evaluate_knn(3, &test_dataset);
  ASSERT_IN_RANGE(0.4, evaluate_knn(3, &test_dataset), FLOAT_TOLERANCE);

  free(test_dataset.points);
  free(test_dataset.categories);

  PASS();
}


//Test that the correct number is returned after a call to the string is passed to the classifier

/* Suites can group multiple tests with common setup. */
SUITE(external_suite) {
    RUN_TEST(distance_1_dimension);
    RUN_TEST(distance_1_dimension_fraction);
    RUN_TEST(distance_3_dimensions);
    RUN_TEST(distance_10_dimensions);

    RUN_TEST(classify_int);

    RUN_TEST(most_frequent_3_inputs);
    RUN_TEST(most_frequent_with_zero);
    RUN_TEST(most_frequent_7_inputs);

    RUN_TEST(compare_ints);
    RUN_TEST(compare_greater_int);
    RUN_TEST(compare_very_different_int_positive);
    RUN_TEST(compare_very_different_int_negative);

    RUN_TEST(find_1_nearest_neighbour);
    RUN_TEST(find_1_nearest_neighbour_parallel);
    RUN_TEST(find_3_nearest_neighbour);
    RUN_TEST(find_3_nearest_neighbour_parallel);

    RUN_TEST(extract_field_1);
    RUN_TEST(extract_field_4);
    RUN_TEST(extract_field_different_formatting);
    RUN_TEST(extract_flower_field);
    RUN_TEST(field_2);
    RUN_TEST(out_of_bounds);

    //Testing the configuration manager
    RUN_TEST(gets_class_int);
    RUN_TEST(initialise_category);
    RUN_TEST(create_first_category);
    RUN_TEST(create_new_category);

    RUN_TEST(knn_accuracy);
}

/* Add definitions that need to be in the test runner's main file. */
>>>>>>> 09c7a9287694984fddbb27faf1b51f6e8306a8b1
GREATEST_SUITE(external_suite);