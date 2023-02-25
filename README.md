# 🤖 Real-time-Object-2-D-Recognition

The objective of the project is to develop an object recognition system capable of identifying a set of specified objects in real-time from a camera looking down onto a white surface. The system should differentiate objects based on their 2D shape and a uniform dark color, regardless of their position, scale, or rotation. By completing each of the five tasks, the project aims to create an efficient and effective object recognition system that could be used in a variety of applications.

If encounter any problems, please feel free to contact us via email hu.hui1@northeastern.edu or zhong.yao@northeastern.edu

## Travel time

Use 1 day of travel time

## Environment

MacOS M1 chip

IDE: VS Code

Build with Makefile

## Links

- link to report: https://cerulean-novel-54b.notion.site/Report-3-Real-time-Object-2-D-Recognition-b63f042b3d934781ad0a714c45e3892f

- link to github: https://github.com/MelancholyZhong/Real-time-Object-2-D-Recognition

- link to demo video: https://youtu.be/O6WMgxuzKnc

## How to run the code

In the treminal

1. Enter command `make vidDisplay` to compile cpp files

2. Then `./vidDisplay` to run the excutable

3. Keypress Instructions:

   - `q`: exit the program
   - `a`: modify thresholded value (in an new window), use `a` to save modification and quit the new window
   - `r`: start or stop recognization
   - `t`: save the training data, after pressing this, the video is paused, you should in put the label in the terminal and after input label, the feature is saved and video is resumed.
   - `m` set the classifier to "nearest neighbor"
   - `n` set the classifier to "3-nearest-neighbor"
   - `s`: save images for the report(used when developing)
   - `l`: only to save the recognied image(used when developing)
