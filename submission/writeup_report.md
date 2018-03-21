# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ../figs/figure_1.png "Original Distribution"
[image2]: ../figs/figure_2.png "Post-processed Distribution"
[image3]: ../figs/figure_3.png "Model MSE Loss"
[image4]: ../figs/figure_4.png "Model Visualization"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 recording implemented model performance on Track 1
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

To speed up the test of model.h5 performance, I modeified drive.py to make car can run in a maximum speed. Due to I run simulator in Win 10 using git base, I added following codes to avoid OSError:
```sh
import win_unicode_console
win_unicode_console.enable()
```

In OSX or Linux, you may not need these lines in your drive.py.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I firstly used a simple convolution neural network to make sure the model can be used for autonomous driving. Then a LeNet is adopted and trained with simple collected data (one round for forwarding driving). Using LeNet structure, the model can drive pretty well and smoothly for straight lane and a few small turns. However, for sharp turns and bridge parts, the model always ran out of roads. Therefore, finally I used the [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) in my codes (model.py lines 108-116). 

![alt text][image4]

I directly tried the model and it had a good result, therefore, I didn't change the structure of the model. Becasue in [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), no details about nomalization layer, I used "input/255.0 - 0.5" in Lambda layer (model.py line 106) and also cropped the upper and lower parts of the input images in Cropping layer (model.py line 107). These two prepossing methods for input images can significantly reduce the noise. 

#### 2. Attempts to reduce overfitting in the model 

I have tried to use dropout layers in order to reduce overfitting, however, the performance is not quite obviously better. Thus in final model, I did not include the dropout layer. 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, counter-clock-wise driving, and some smooth curves driving. Besides, I used all three cameras' images to increase the size of training data (model.py line 70-88).  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Model Design

The overall strategy for deriving a model architecture was to build a convolution neural network that processes the input cameras' images to properly predict the steering angle of the vehicle.  

I used a well-known convolution neural network structure (nVidia model), this is because we can learn from others, save time, and probably improve the performance based on other's excellent work. 

To improve the performance, as I mentioned, I added a Lambda layer and a cropping layer to the model. However, the model cannot finish a satisfied lap in autonomous mode. After read other students' suggestions, I reallized that there were some limitations in my data set, i.e., a bias towoards to zero, more left turn than right turn. Therefore, I processed on the data set with augmentation and distribution modification. 


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two clock-wise laps and two counter-clock-wise laps on Track 1 using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to steer back from road side to center. I found the vehicle always ran out of road at turns, to make the model better at turns, I recorded more data only at turns. 

After the collection process, I ploted the distribution of the data (model.py line 27-34) and verified the problem of the bias to zero.

![alt text][image1]

To tailor the zero steering angle inputs into a reasonable size, I set the target size to be 0.7 x average samples per bin (model.py line 37). Then, for each bin whose number of inputs, I reduced the size to the target size (model.py line 38-50). The post-processed distribution is in a Gaussian-like shape.

![alt text][image2]

To balance the negative and positive steering angles, I flipped the input with a 50% probability (model.py line 83-85). A generator was built to avoid overloading the memory (model.py line 60-93).  

I finally randomly shuffled the data set and put 10% of the data into a validation set. After all data processing, I had 17712 total inputs, 15939 training inputs, and 1773 validation inputs.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I observed that usually after 3 epochs, the loss would increase. Therefore, I chose 3 as the number of epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary. The model MSE loss is illustrated.

![alt text][image3]

You can check the final model performance in [run1.mp4](./)
