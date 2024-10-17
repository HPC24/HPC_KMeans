#ifndef UTILS_H
#define UTILS_H

#ifdef USE_CONT_MEM
#include <Cont_Mem_Parallel_KMeans.h>
#else
#include <Parallel_KMeans.h>
#endif

#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib> 
#include <algorithm>
#include <cctype> 
#include <sstream>
#include <zlib.h>
#include <cstring>
#include <KMeans.h>
#include <omp.h>



template <typename FType> 
void CenterImages(std::vector<std::vector<FType>>& data, int image_size){

    const std::size_t cols = data.empty() ? 0 : data[0].size();
    float pixel_average = 0.0; 
    std::size_t total_pixel_values = image_size * data.size();
    double pixel_sum = 0;
    
    #pragma omp parallel for collapse(2) default(none) shared(data, cols) reduction( +: pixel_sum)
    for (std::size_t image = 0; image < data.size(); ++image)
    {
        for (std::size_t col_idx = 0; col_idx < cols; ++col_idx)
        {
            pixel_sum += data[image][col_idx];
        }
    }

    if (total_pixel_values > 0) {
        pixel_average = pixel_sum / total_pixel_values;
    }
    else
    {
        std::cout << "Total Pixel Values <= 0 check the data" << std::endl;
    }

    // modify the data and subtract the average from all the pixel values

    #pragma omp parallel for collapse(2) default(none) shared(data, cols, pixel_average) 
    for (std::size_t image = 0; image < data.size(); ++image)
    {
        for (std::size_t col_idx = 0; col_idx < cols; ++col_idx)
        {
            data[image][col_idx] /= pixel_average;
        }
    }


}


template <typename FType>
std::pair<std::vector<std::vector<FType>>, std::vector<int>> GenerateTestData(int n, int cols, int n_cluster, double mean, double stddev) {

    std::vector<int> labels(n * n_cluster, 0);
    std::vector<std::vector<FType>> points;
    points.resize(n * n_cluster); 
    for (int cluster = 0; cluster < n_cluster; ++cluster)
    {
        std::random_device rd;  
        std::mt19937 gen(rd()); 

        // every cluster the cennter of the points is shifted by 5
        std::normal_distribution<> d(mean + (cluster * 4), stddev);

        for (int i = 0; i < n; ++i)
        {

            int index_points = i + cluster * n;
            points[index_points].resize(cols);
            labels[index_points] = cluster;

            for (int j = 0; j < cols; ++j)
            {
                points[index_points][j] = d(gen);
            }
            
        }
    }

    return std::make_pair(points, labels);
}


template <typename FType, typename IType = std::size_t>
std::tuple<std::vector<double>, std::vector<int>, double, double, double> TimeKMeans(
    const int n_cluster, 
    const int max_iter, 
    const double tol, 
    const int seed, 
    const int iterations, 
    const std::vector<std::vector<FType>>& data){

    std::vector<double> KMeans_timings(iterations, 0.0);
    std::vector<int> KMeans_iterations(iterations, 0);
    std::cout << "Start timing KMeans for " << iterations << " Iterations" << std::endl;

    for (int iteration = 0; iteration < iterations; ++iteration)
    {

        KMeans<FType, IType> kmeans(n_cluster, max_iter, tol, seed);

        auto start = std::chrono::high_resolution_clock::now();

        kmeans.fit(data);

        auto end = std::chrono::high_resolution_clock::now();
   
        std::chrono::duration<double, std::milli> elapsed = end - start;
        KMeans_timings[iteration] = elapsed.count();
        KMeans_iterations[iteration] = kmeans.n_iter;
    }

    double KMeans_average = 0.0;
    double KMeans_min = KMeans_timings[0];
    double KMeans_max = KMeans_timings[0];

    for (int i = 0; i < KMeans_timings.size(); ++i)
    {
        KMeans_average += KMeans_timings[i];

        if (KMeans_timings[i] < KMeans_min)
        {
            KMeans_min = KMeans_timings[i];
        }

        if (KMeans_timings[i] > KMeans_max)
        {
            KMeans_max = KMeans_timings[i];
        }

    }

    KMeans_average /= KMeans_timings.size();

    return std::make_tuple(KMeans_timings, KMeans_iterations, KMeans_average, KMeans_min, KMeans_max);

}

template <typename FType, typename IType = std::size_t>
std::tuple<std::vector<double>, std::vector<int>, double, double, double> TimeParallelKMeans(
    const int n_cluster, 
    const int max_iter, 
    const double tol, 
    const int seed,  
    int iterations, const std::vector<std::vector<FType>>& data){

    std::vector<double> KMeans_timings(iterations, 0.0);
    std::vector<int> KMeans_iterations(iterations, 0);
    std::cout << "Start timing Parallel KMeans for " << iterations << " Iterations" << std::endl;

    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        Parallel_KMeans<FType, IType> kmeans(n_cluster, max_iter, tol, seed);
        auto start = std::chrono::high_resolution_clock::now();

        kmeans.fit(data);

        auto end = std::chrono::high_resolution_clock::now();
   
        std::chrono::duration<double, std::milli> elapsed = end - start;
        KMeans_timings[iteration] = elapsed.count();
        KMeans_iterations[iteration] = kmeans.n_iter;
    }

    double KMeans_average = 0.0;
    double KMeans_min = KMeans_timings[0];
    double KMeans_max = KMeans_timings[0];

    for (int i = 0; i < KMeans_timings.size(); ++i)
    {
        KMeans_average += KMeans_timings[i];

        if (KMeans_timings[i] < KMeans_min)
        {
            KMeans_min = KMeans_timings[i];
        }

        if (KMeans_timings[i] > KMeans_max)
        {
            KMeans_max = KMeans_timings[i];
        }

    }

    KMeans_average /= KMeans_timings.size();

    return std::make_tuple(KMeans_timings, KMeans_iterations, KMeans_average, KMeans_min, KMeans_max);

}

#endif