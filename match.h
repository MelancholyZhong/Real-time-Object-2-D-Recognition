/**
    Real-time Object 2-D Recognition
    Created by Yao Zhong for CS 5330 Computer Vision Spring 2023

    Functions related to distances and classifiers.
*/

#ifndef MATCH
#define MATCH

//Classifier 1: nearest neighbor
int nearestNeighbor(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label);

//Classifier 2: 3-nearest neigbor.(this classifier requires at least 3 examples of that object)
int nearest3(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label);

#endif