#ifndef TESTS_H
#define TESTS_H

#include <vector>

void CheckLabels();
template <typename FType>
void CheckData(std::vector<std::vector<FType>>& data);
#endif