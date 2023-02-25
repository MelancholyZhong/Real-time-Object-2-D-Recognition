/**
    Real-time Object 2-D Recognition
    Created by Yao Zhong for CS 5330 Computer Vision Spring 2023

    Functions related to distances and classifiers.
*/

#include <iostream>
#include <cmath>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "match.h"

//Calculate the scaled Euclidean Distance of two features: |x1-x2|/st_dev
float scaledEuclideanDis(std::vector<float> &feature1, std::vector<float> &feature2, std::vector<float> &deviations){
    float distance = 0.0;
    for(int i=0; i<feature1.size(); i++){
        distance += std::sqrt((feature1[i] - feature2[i])*(feature1[i] - feature2[i])/deviations[i]);//here is the square root
    }
    return distance;
}

//Calculate the standard deveation for each entry of the features in the database(for sacle), the result is not square rooted until next function.
int standardDeviation(std::vector<std::vector<float>> &data, std::vector<float> &deviations){
    std::vector<float> sums = std::vector<float>(data[0].size(), 0.0); //sum of each entry
    std::vector<float> avgs = std::vector<float>(data[0].size(), 0.0); //average of each entry
    std::vector<float> sumSqure = std::vector<float>(data[0].size(), 0.0); //sum of suqared difference between x_i and x_avg
    deviations = std::vector<float>(data[0].size(), 0.0); //final result(deviations not square rooted yet)

    //first loop for the sum of each entry
    for(int i=0; i<data.size(); i++){
        for(int j=0; j<data[0].size();j++){
            sums[j] += data[i][j];
        }
    }

    //calculate the avgs
    for(int i=0; i<sums.size(); i++){
        avgs[i] = sums[i]/data.size(); //average
    }

    //second loop, for the sum of  suqared difference of  each entry
    for(int i=0; i<data.size(); i++){
        for(int j=0; j<data[0].size();j++){
            sumSqure[j] += (data[i][j]-avgs[j])*(data[i][j]-avgs[j]);
        }
    }

    //the deviations
    for(int i=0; i<sumSqure.size(); i++){
        deviations[i] = sumSqure[i]/(data.size()-1); 
    }

    return 0;
}


//Classifier 1: nearest neighbor
int nearestNeighbor(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label){
    std::vector<float> deviations;
    standardDeviation(data,deviations);
    float minDis = -1;
    for(int i=0; i<data.size(); i++){
        float dis = scaledEuclideanDis(feature, data[i], deviations);
        if(minDis == -1 || dis<= minDis){
            minDis = dis;
            std::strcpy(label, labels[i]);
        }
    }
    return 0;
}

//Classifier 2: 3-nearest neigbor.(this classifier requires at least 3 examples of that object)
int nearest3(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label){
    std::vector<float> deviations;
    standardDeviation(data,deviations);
    //calcuted distances and store by label
    std::unordered_map<std::string, std::vector<float>> map;
    for(int i=0; i<data.size(); i++){
        float dis = scaledEuclideanDis(feature, data[i], deviations);
        std::string slabel(labels[i]);
        //https://stackoverflow.com/questions/69456226/class-stdmap-has-no-member-contains-in-visual-studio-code
        if(map.find(slabel) == map.end()){
            std::vector<float> newVec;
            newVec.push_back(dis);
            std::pair<std::string , std::vector<float>> objectDis (slabel, newVec);
            map.insert(objectDis);
        }else{
            map[slabel].push_back(dis);
        }
    }

    //get each label's  top 3 and then get the best match label
    float minDis = -1;
    //https://stackoverflow.com/questions/26281979/c-loop-through-map
    for(auto const& x: map){
        // std::cout<<x.first<<std::endl;
        std::vector<float> distances = x.second;
        if(distances.size()<3){
            continue;
        }
        std::sort(distances.begin(), distances.end());
        float distance = 0.0;
        for(int i=0; i<3; i++){
            distance += distances[i];
        }
        distance = distance/3;
        if(minDis == -1 || distance< minDis){
            minDis = distance;
            std::strcpy(label, x.first.c_str());
        }
    }

    return 0;
}

