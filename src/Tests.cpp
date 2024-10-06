#include <Tests.h>
#include <Parallel_KMeans.h>
#include <KMeans.h>
#include <vector>
#include <iostream>

const double TOL = 1e-9;
const int MAX_ITER = 500;

void CheckLabels(){

    std::vector<std::vector<float>> test_data =   {{1.2, 1.5, 1.8},
                                                    {3.5, 3.6, 3.8},
                                                    {1.3, 1.4, 1.9},
                                                    {3.4, 3.5, 3.9},
                                                    {10.2, 10.1, 10.4},
                                                    {10.5, 10.6, 10.9}
                                                    };

    std::vector<int> test_labels = {0, 1, 0, 1, 2, 2};

    const int N_CLUSTER = 3;

    KMeans<float, std::size_t> kmeans(N_CLUSTER, MAX_ITER, TOL);
    Parallel_KMeans<float, std::size_t> parallel_kmeans(N_CLUSTER, MAX_ITER, TOL);

    std::cout << "Beginning fitting for KMeans:" << std::endl;
    kmeans.fit(test_data);

    std::cout << "Beginning fitting for Parallel KMeans:" << std::endl;
    parallel_kmeans.fit(test_data);


    std::cout << "Test Labels:" << std::endl;

    for (int i = 0; i < test_labels.size(); ++i)
    {
        std::cout << test_labels[i] << " ";
    }

     std::cout << std::endl;

    std::cout << "KMeans Fit Labels" << std::endl;

    for (int i = 0; i < kmeans.labels.size(); ++i)
    {
        std::cout << kmeans.labels[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "Parallel KMeans Fit Labels" << std::endl;

    for (int i = 0; i < parallel_kmeans.labels.size(); ++i)
    {
        std::cout << parallel_kmeans.labels[i] << " ";
    }

    std::cout << std::endl;


}

template <typename FType>
void CheckData(std::vector<std::vector<FType>>& data){

    const std::size_t ROWS = data.size();
    const std::size_t COLUMNS = data[0].size();

    std::cout << "First 3 images:" << std::endl;
    for (int i = 0; i < 3 ; ++i)
    {
        for (int j = 0; j < COLUMNS; ++j)
        {
            std::cout << data[i][j] << ",";
        }

        std::cout << std::endl;
    }

    std::cout << "Last 3 images:" << std::endl;
    std::vector<int> counters;
    for (int i = ROWS - 3; i < ROWS; ++i)
    {
        int counter = 0;
         for (int j = 0; j < COLUMNS; ++j)
        {
            std::cout << data[i][j] << " ";
            counter += 1;
        }
        
        counters.push_back(counter);

        std::cout << std::endl;
    }

    for (int i = 0; i < counters.size(); ++i)
    {
        std::cout << "Counter: " << counters[i] << std::endl;
    }

}





