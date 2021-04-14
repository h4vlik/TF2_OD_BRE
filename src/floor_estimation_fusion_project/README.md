# Floor detection during elevator ride by Bayes fusion

Main goal of this task is to detect actual floor during elevator ride. This task is esencial for mobile robot to be able to move in multifloor buildings with elevator. 

Hardware:
Camera - Microsoft LifeCam HD 3000
IMU sensor - MPU 9250 with acceloremeter, gyroscope, compass, thermometer

Solution:
1. use acceleration data
    - from acceleration compute displacement
    - eliminate DC bias
    - universal usage - it will calibrate to elevator, that is use to ride

2. use camera data
    - use data from camera, that shows information display with floor numbers
    - train Convolution Neural Network (CNN) to classify pictures

3. fuse information from algorithm
    - data fusion by Bayesian filtr


## Folders
1. core
2. input_feed
3. main
4. App.py

## Parts 

### Floor detection via accelerometr algoritm
input - acc data
output - floor difference in real time

### Floor detection via camera
input - fram from camera
output - actual floor 

### Data fusion using Bayes filter
input - floor difference, actual floor
ouptu - floor result 
