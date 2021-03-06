#**Behavioral Cloning** 



**Behavioral Cloning Project**


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior

* Build, a convolution neural network in Keras that predicts steering angles from images

* Train and validate the model with a training and validation set

* Test that the model successfully drives around track one without leaving the road

* Summarize the results with a written report





[//]: # (Image References)



[image1]: ./examples/placeholder.png "Model Visualization"

[image2]: ./examples/placeholder.png "Grayscaling"

[image3]: ./examples/placeholder_small.png "Recovery Image"

[image4]: ./examples/placeholder_small.png "Recovery Image"

[image5]: ./examples/placeholder_small.png "Recovery Image"

[image6]: ./examples/placeholder_small.png "Normal Image"

[image7]: ./examples/placeholder_small.png "Flipped Image"



## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  



---

###Files Submitted & Code Quality



####1. Submission includes all required files and can be used to run the simulator in autonomous mode



My project includes the following files:

* model.py containing the script to create and train the model

* drive.py for driving the car in autonomous mode

* model.h5 containing a trained convolution neural network 

* writeup_report.md or writeup_report.pdf summarizing the results



####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh

python drive.py model.h5

```



####3. Submission code is usable and readable



The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy



#### 1. An appropriate model architecture has been employed


My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 32 and 100 (model.py lines 116-124)

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer (code line 115).



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 126, 128). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 103). I defined a generator to argument the training image(line 54-101). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used a combination of center lane driving, recovering from the left and right sides of the road. For the images comes from left side, I increase the corresponding steering by 0.07 which means make it turn left more. And for the right-side image I make it turn left more.(code line 64-79 )

And I randomly flip the training images, and multiply the steering angle by -1 when the images were flipped obviously.

For details about how I created the training data, see the next section.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to design the basic model, train the model and use simulator the see how does it work. Then upgrade the model based on the previous models repeatly.


My first step was to use a convolution neural network model similar to the <End-to-End Deep Learning for Self-Driving Cars> published by nvidia. I thought this model might be appropriate because we face the same problem, and the parameters number it has is not large, which reduced the risk of overfitting.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by adding two dropout layers to the fully-connected layers. And I use the left side and right side images of the camera, to increase the training data number. And I argumenting the input image by randomly change the brightness and contract of the images and randomly flip the images.

Then I use jupyte notebook to write a program to find out the distribution of steering angle data (in Visualize data.ipynb).

![](img/distribution.png)

From the image we can find out that the images whose  corresponding steering angle between -0.1 and 0.1 is majority, the number of these images is more than the half of the total images number. 

![](img/steering.png)

What's more, we can see that even though these images has different steering angle, they are pretty mush the same, thus these data may introduce too much noise which we don't want. So I chose to remove most of them(code lines 35-47) to get a more reliable training data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I collect more data in these tricky spots, and retrained the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.



####2. Final Model Architecture

The final model architecture (model.py lines 113-131) consisted of a convolution neural network with the following layers and layer sizes ...


Here is a visualization of the architecture

![](img/architecture.png)


####3. Creation of the Training Set & Training Process



To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


![](img/center_2017_05_24_11_21_28_388.jpg)


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recorvering from bad state. These images show what a recovery looks like starting from:

![](img/recovering.png)

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add more valid data.

After the collection process, I had 48433 number of data points. I then preprocessed this data by removing most of the images which steering angle is small. And I randomly change the brightness and contract of the images(code lines 82-87 ).


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60 as evidenced by the plot image:

![](img/los.png)

I used an adam optimizer so that manually training the learning rate wasn't necessary.

