#include <KMeans.h>
#include <Parallel_KMeans.h>
#include <utils.h>
#include <Tests.h>

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
#include <omp.h>
#include <filesystem>

using IMAGE_DATA_TYPE = float;
using ITYPE = std::size_t;

// Image size of the MNIST data set
const int IMAGE_SIZE = 28 * 28;

// Parameters for the KMeans Algorithm
const double TOL = 1e-9;
const int N_CLUSTER = 10;
const int MAX_ITER = 500;

// how many timings are being done
const int TIMING_ITERATIONS = 20;

// Parameters to generate the Test data
const int N = 2000;
const int COLS = 500;
const int STD = 8;

// Seed that is used to time the KMeans and Parallel KMeans implementation
const int SEED = 42;


void print_usage() {

    std::cout << "Usage: program --data <filepath> --output <filepath> [--verbose]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "--data <value>        Provide data input" << std::endl;
    std::cout << "--output <filepath>  Specify output file" << std::endl;
    std::cout << "--verbose             Enable verbose mode" << std::endl;
}

bool endsWithGZ(const std::string& path) {

    const std::string extension = ".gz";

    // Check if the path is at least as long as the extension
    if (path.length() >= extension.length()) 
    {
        // Extract the last 3 characters and compare with ".gz"
        return path.compare(path.length() - extension.length(), extension.length(), extension) == 0;
    }

    // If the string is too short, it can't have the extension
    return false;
}

template <typename FType>
void readImagesAndLabelsGzip(const char* filename, std::vector<std::vector<float>> &images, std::vector<int> &labels, int image_size) {

    gzFile gz_file = gzopen(filename, "rb");  // Open the gzipped file in binary mode
    if (!gz_file) {
        std::cerr << "Error: Cannot open gzipped file " << filename << std::endl;
        exit(1);
    }

    // Buffer size is 4096 because max chars in line = 784 (IMAGE_SIZE) * 3 (at most 3 chars for pixel values) + 784 (commas) + 1 (label) + 2 (newline and \0)
    // so at most = 3139 bytes
    const int buffer_size = 4096; 
    char buffer[buffer_size];


    size_t image_count = 0;
    const char* start = "@data";
    bool found_start = false;

    std::cout << "Start processing the images" << std::endl;

    while (gzgets(gz_file, buffer, buffer_size)) {
        
        if (!found_start)
        {
        for (const char* ptr = buffer; *ptr != '\0'; ++ptr)
        {
            bool match = true;
            for (int i = 0; i < std::strlen(start); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(ptr[i])) != std::tolower(static_cast<unsigned char>(start[i])))
                {
                    match = false;
                    break;
                }
            }

            if (match == true)
            {
                found_start = true;
                break;
            }
        }
        }
        else
        {
            std::vector<FType> image (image_size, 0);
            std::string tmp;
            std::size_t counter = 0;
            // Read pixel values for the image
            for (const char* ch = buffer; *ch != '\0'; ++ch){
            
                if (*ch == ',')
                {
                    image[counter] = std::stof(tmp);
                    tmp.clear();
                    counter += 1;
                }
                else if (*ch == '\n')
                {
                    continue;
                }
                else {
                    tmp.push_back(*ch);
                }
            }


            if (!tmp.empty())
            {
                labels.push_back(std::stof(tmp));
            }

            if (image.empty())
            {
                std::cerr << "image is empty" << std::endl;
            }
            
            // sanity check to ensure that all the images have been read correctly
            if (image.size() != image_size)
            {
                std::cout << "Image size " << image.size() << std::endl;
                std::cout << "size of image "<< image_count << " is not correct" << std::endl;
            }


            images.push_back(image); 
            image_count += 1;


        }

    }

    std::cout << "Image processing has ended" << std::endl;
    std::cout << "The Image count is " << image_count << std::endl;
    
    if (gzclose(gz_file) != Z_OK) {
    std::cerr << "Error: Failed to close the gzipped file" << std::endl;
    return;
    }

    std::cout << "File closed successfully." << std::endl;
}


template <typename FType>
void readImagesAndLabels(const std::string &filename, std::vector<std::vector<float>> &images, std::vector<int> &labels, int image_size) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    std::string line;

    std::cout << "Opening the file" << std::endl;
    // Find the @Data marker
    while (std::getline(file, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line == "@Data" || line == "@data" || line == "@DATA") {
            std::cout << "Found @Data line" << std::endl;
            break;
        }
    }

    std::size_t image_count = 0;
    // Read the images and labels data
    std::cout << "Start processing the images" << std::endl;
    while (std::getline(file, line)) {
        if (line.empty()) 
        {
            continue; // Skip empty lines
        }

        else
        {
            std::vector<FType> image (image_size, 0);
            size_t counter = 0;
            std::string tmp;

            // Read pixel values for the image
            for (char ch: line)
            {
                
                if (ch == ',')
                {
                    
                    image[counter] = std::stof(tmp);
                    tmp.clear();
                    counter += 1;
                }
                else 
                {
                    tmp.push_back(ch);
                }
            }

            // the last value which is the label is not pushed into the image vector 
            // because the lookback value is only saved if a comma i read after the value which is not the case for the last 
            // value that is the label
            if (!tmp.empty())
            {
                labels.push_back(std::stoi(tmp));
            }
        
            if (image.empty())
            {
                std::cerr << "image is empty" << std::endl;
            }
            
            // sanity check to ensure that all the images have been read correctly
            else if (image.size() != image_size)
            {
        
                std::cout << "size of image "<< image_count << " is not correct" << std::endl;
            }

            images.push_back(image); 
            image_count += 1;
        }
    }


    std::cout << "The Image count is " << image_count << std::endl;
    std::cout << "Image processing has ended" << std::endl;

    file.close();
}

int main(int argc, char* argv[]){

    if (argc == 1)
    {
        print_usage();
        return 1; 
    }

    std::vector<std::vector<float>> data;
    std::vector<int> labels;
    std::string filename;
    std::string output_file;
    bool verbose = false;

    // check if all required arguments have values
    bool has_data = false;
    bool has_output = false;

    // check wheter all the arguments are used and correctly specified
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--data")
        {
            if(i + 1 < argc && argv[i + 1][0] != '-')
            {
                if (std::filesystem::exists(argv[++i]))
                {
                    filename = argv[i];
                    std::cout << "Data File Path: " << filename << std::endl;
                    has_data = true;
                }
                else
                {
                    std::cerr << "--data filepath does not exist: " << arg << std::endl;
                    return 1; 
                }

            }
            else 
            {
                std::cerr << "--data Path requires a valied value" << std::endl;
                print_usage();
                return 1;
            }
        }

        else if (arg == "--output")
        {
            if (i + 1 << argc && argv[i + 1][0] != '-')
            {
                if (std::filesystem::exists(argv[++i]))
                {
                    output_file = argv[i];
                    std::cout << "Output File Path: " << output_file << std::endl;
                    has_output = true;
                }
                else
                {
                    std::cerr << "--output filepath does not exist: " << arg << std::endl; 
                    return 1;
                }
            }
            else 
            {
                std::cerr << "--output Path requires a valied value" << std::endl;
                print_usage();
                return 1;
            }

        }
        else if (arg == "--verbose")
        {
            std::cout << "Verobse ENABLED" << std::endl;
            verbose = true;
        }
        else 
        {
            std::cerr << "Unknown Argument" << std::endl;
            std::cout << "Number of arguments: " << argc << std::endl;
            for (int i = 0; i < argc; ++i) 
            {
                std::cout << "Argument " << i << ": " << argv[i] << std::endl;
            }
            print_usage();
            return 1;
        }
    }

    if (!has_data)
    {
        std::cerr << "No data file has been specified" << std::endl;
        print_usage();
        return 1;
    }
    else if (!has_output)
    {
        std::cerr << "No output file has been specified" << std::endl;
        print_usage();
        return 1;
    }


    const char* omp_threads_env = std::getenv("OMP_NUM_THREADS");
    int omp_num_threads;

    if (omp_threads_env != nullptr)
    {
        omp_num_threads = std::stoi(omp_threads_env);
    }
    else 
    {
        omp_num_threads = 0;
        std::cout << "OMP_NUM_THREADS is not set." << std::endl;
    }
    
    std::cout << "OMP_NUM_THREADS: " << omp_num_threads << std::endl;
    std::cout << "Number of Iterations for timing: " << TIMING_ITERATIONS << std::endl;


    if (endsWithGZ(filename))
    {
        std::cout << "Parsing gzipped file" << std::endl;
        const char* c_str_filename = filename.c_str();
        readImagesAndLabelsGzip<IMAGE_DATA_TYPE>(c_str_filename, data, labels, IMAGE_SIZE);
    }

    else
    {
        readImagesAndLabels<IMAGE_DATA_TYPE>(filename, data, labels, IMAGE_SIZE);
    }

    //CheckLabels();

    // std::cout << "Generating Test Data" << std::endl;

    // auto [test_data, test_labels] = GenerateTestData<IMAGE_DATA_TYPE>(N, COLS, N_CLUSTER, 1.0, STD);
    
    // std::cout << "Expected Test Data Size:" << N * N_CLUSTER << std::endl;
    // std::cout << "Actual Test Data Size:" << test_data.size() << std::endl;
    // std::cout << "Expected Labels Size:" << N * N_CLUSTER << std::endl;
    // std::cout << "Actual Labels Size:" << test_labels.size() << std::endl;
    

    // auto [KMeans_timings, KMeans_iterations, KMeans_average, KMeans_min, KMeans_max] = TimeKMeans<IMAGE_DATA_TYPE, std::size_t>(N_CLUSTER, 
    //                                                                                                                             MAX_ITER, 
    //                                                                                                                             TOL, 
    //                                                                                                                             SEED, 
    //                                                                                                                             TIMING_ITERATIONS, 
    //                                                                                                                             data);

    auto [Parallel_KMeans_timings, Parallel_KMeans_iterations, Parallel_KMeans_average, Parallel_KMeans_min, Parallel_KMeans_max] = TimeParallelKMeans<float, std::size_t>(N_CLUSTER, 
                                                                                                                                                                           MAX_ITER, 
                                                                                                                                                                           TOL,
                                                                                                                                                                           SEED, 
                                                                                                                                                                           TIMING_ITERATIONS, 
                                                                                                                                                                           data);

    // std::cout << "Kmeans all timings in milliseconds" << std::endl;

    // for (int i = 0; i < KMeans_timings.size(); ++i)
    // {
    //     std::cout << KMeans_timings[i] << " ";
    // }

    // std::cout << std::endl;

    // std::cout << "KMeans Average: " << KMeans_average << " ms" << std::endl;
    // std::cout << "KMeans Min: " << KMeans_min << " ms" << std::endl;
    // std::cout << "KMeans Max: " << KMeans_max << " ms" << std::endl;

    std::cout << "Parallel Kmeans all timings in milliseconds" << std::endl;

    for (int i = 0; i < Parallel_KMeans_timings.size(); ++i)
    {
        std::cout << Parallel_KMeans_timings[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "Parallel KMeans Average: " << Parallel_KMeans_average << " ms" << std::endl;
    std::cout << "Parallel KMeans Min: " << Parallel_KMeans_min << " ms" << std::endl;
    std::cout << "Parallel KMeans Max: " << Parallel_KMeans_max << " ms" << std::endl;


    std::cout << "Output File Name: " << output_file << std::endl;


    // Open the file in read mode
    std::ifstream File(output_file, std::ios::ate);

    if (!File.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        return 1;
    }

    // Check current position. Before cursor was placed at the end so if there would 
    // be any content in the file the position would be != 0 
    std::streampos filesize = File.tellg();  
    File.close();

    // std::ios::app makes it possible that i can add lines to an already existing file
    // write the output to the file in the following format 
    // COLUMNS: OMP_NUM_THREADS TIMINGS ITERATIONS
    if (filesize == 0)
    {
        std::ofstream outFile(output_file, std::ios::binary);
        if (outFile.is_open())
        {
            if (verbose)
            {
                std::cout << "OMP_NUM_THREADS" << "\t" << "FIT_TIME" << "\t" << "NUM_ITERATIONS" << std::endl;
                outFile << "OMP_NUM_THREADS" << "\t" << "FIT_TIME" << "\t" << "NUM_ITERATIONS" << std::endl;
                for (int i = 0; i < TIMING_ITERATIONS; ++i)
                {
                    std::cout << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                    outFile << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                }
            }
            else
            {
                outFile << "OMP_NUM_THREADS" << "\t" << "FIT_TIME" << "\t" << "NUM_ITERATIONS" << std::endl;
                for (int i = 0; i < TIMING_ITERATIONS; ++i)
                {
                    outFile << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                }
            }


            outFile.close();
        }
        else
        {
              std::cerr << "Unable to open file" << std::endl;
        }
    }
    else
    {
        std::ofstream outFile(output_file, std::ios::app | std::ios::binary);
        if (outFile.is_open()) 
        {

            if (verbose)
            {

                for (int i = 0; i < TIMING_ITERATIONS; ++i)
                {
                    std::cout << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                    outFile << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                }
            }
            else
            {
                for (int i = 0; i < TIMING_ITERATIONS; ++i)
                {
                    outFile << omp_num_threads << "\t" << Parallel_KMeans_timings[i] << "\t" << Parallel_KMeans_iterations[i] << std::endl;
                }
            }

            outFile.close();

        } 
        else 
        {
            std::cerr << "Unable to open file" << std::endl;
        }
    }

    return 0;

    // // Number of iterations for timing
    // const int numIterations = 500;

    // // File to write timing results
    // std::ofstream outputFile("timing_results.txt");

    // // Check if the file opened successfully
    // if (!outputFile.is_open()) {
    //     std::cerr << "Failed to open output file!" << std::endl;
    //     return 1;
    // }

    // std::vector<std::vector<double>> test_data =   {{1.2, 1.5, 1,8},
    //                                                 {3.5, 3.6, 3.8},
    //                                                 {1.3, 1.4, 1.9},
    //                                                 {3.4, 3.5, 3.9},
    //                                                 {10.2, 10.1, 10.4},
    //                                                 {10.5, 10.6, 10.9}
    //                                                 };

    // std::vector<int> test_labels = {0, 1, 0, 1, 2, 2};


    // #ifdef USE_KMEANS

    // #elif defined(USE_PARALLEL_KMEANS)

     

    // #endif

    

}