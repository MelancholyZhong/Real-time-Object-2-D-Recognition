#include <iostream>
#include <cmath>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

#include "match.h"


float euclideanDis(std::vector<float> &feature1, std::vector<float> &feature2){
    float distance = 0.0;
    for(int i=0; i<feature1.size(); i++){
        distance += (feature1[i] - feature2[i])*(feature1[i] - feature2[i]);
    }
    return distance;
}

int nearestNeighbor(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label){
    float minDis = -1;
    for(int i=0; i<data.size(); i++){
        float dis = euclideanDis(feature, data[i]);
        if(minDis == -1 || dis<= minDis){
            minDis = dis;
            std::strcpy(label, labels[i]);
        }
    }
    return 0;
}

//https://stackoverflow.com/questions/4157687/using-char-as-a-key-in-stdmap
struct cmp_str
{
   bool operator()(char const *a, char const *b) const
   {
      return std::strcmp(a, b) < 0;
   }
};

int nearest3(std::vector<char *> &labels, std::vector<std::vector<float>> &data, std::vector<float> &feature, char* label){
    std::unordered_map<char*, std::vector<float>, cmp_str> map;
    for(int i=0; i<data.size(); i++){
        float dis = euclideanDis(feature, data[i]);
        //https://stackoverflow.com/questions/69456226/class-stdmap-has-no-member-contains-in-visual-studio-code
        if(!map.count(labels[i])){
            std::cout<<dis<<std::endl;
            std::vector<float> newVec;
            newVec.push_back(dis);
            std::pair<char* , std::vector<float>> objectDis (labels[i], newVec);
        }else{
            map[labels[i]].push_back(dis);
            std::cout<<labels[i]<<std::endl;
        }
    }

    float minDis = -1;
    //https://stackoverflow.com/questions/26281979/c-loop-through-map
    for(auto const& x: map){
        std::vector<float> distances = x.second;
        std::sort(distances.begin(), distances.end());
        float distance = 0.0;
        for(int i=0; i<3; i++){
            distance += distances[i];
        }
        distance = distance/3;
         if(minDis == -1 || distance<= minDis){
            minDis = distance;
            std::strcpy(label, x.first);
        }
    }

    return 0;
}

