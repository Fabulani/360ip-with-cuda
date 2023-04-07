#ifndef UTILS_H
#define UTILS_H

#include <iostream>

std::string get_current_time_string();

std::string removeFileExtension(const std::string& filename);

std::string removePath(std::string filepath);

#endif