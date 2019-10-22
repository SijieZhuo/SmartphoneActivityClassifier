# SmartphoneActivityClassifier
SmartphoneActivityClassifier is a python application that can collect real-time smartphone motion sensor data,
performing feature extraction and real-time classification for smartphone activities.

## Software Installation
the latest version of the application can be found in the release page.

1. Extract the file
2. Open the application in any IDE (optional)
3. Run the Main.py

## Functionalities

The main purpose of this application at this stage is collecting and storing data collected from the Android app we developed,
and performing feature extraction and classification

The data collection and real-time classification requires the phone to be paired with the machine that runs this application. 
But feature extraction and other functionalities included in the classification page does not require smartphone pairing.


(note: the data collection page would only be available when the application is paired with a smartphone)

### Data Collection Page
The data collection process is via bBluetooth from an Android app we developed https://github.com/LucasSherlock/ClassifySmartphoneActivity.

The data send from the Android app includes 11 columns of data, which includes 11 columns of data: the time recorded, 
3 axis signal from accelerometer, gyroscope and magnetometer, and the smartphone activity.

- time : the timestamp when the data was collected
- accX : accelerometer signal data for the X axis
- accY : accelerometer signal data for the Y axis
- accZ : accelerometer signal data for the Z axis
- rotX : gyroscope signal data for the X axis
- rotY : gyroscope signal data for the Y axis
- rotZ : gyroscope signal data for the Z axis
- graX : magnetometer signal data for the X axis
- graY : magnetometer signal data for the Y axis
- graZ : magnetometer signal data for the Z axis
- activity : the smartphone activity recorded from the phone

The data would be saved in a folder according to the current time, and each activity will be saved into a separate .csv file.
The application is used to collect 14 different types of smartphone activities.

The recording process would start once the "start recording" button is been pressed (create the folder). At this stage, the 
data is not saved by the application, only when the Android sends the start message, then the data would be recorded.
when The application is recording an activity, the activity label would turn blue, indicating that the application is recording 
the data, once an activity is completed, the label would turn green.

### Feature Extraction and Reduction Page
SmartphoneActivityClassifier can perform two different types of feature extraction:
- using existing library __*tsfresh*__ to automatically perform feature extraction and reduction
- manually select features to extract.

To perform feature extraction and reduction, the data has to be manually placed in the folder called "DataForAnalysation".
the application would read all the csv files 

#### Using __*tsfresh*__
The time required to perform feature extraction and reduction using __*tsfresh*__ library is significantly larger than using
the manual select method.

When using this tsfresh method, the application would first extract all possible features in the library, and save the data
in a csv file, then it would use a relecance table to reduce the number of features. The application would save the reduced 
features in another csv file, and it would be used for real-time classification or re-extract feature for training data.

Due to the large number of features extracted, and each type of data (such as accX, accY) may have different features after 
reduction, the final feature extraction for real-time classification is done separatly for individual type of data 
(therefore, 9 times). So it is not recommended for real-time classification.

#### Manually select features
This method requires developer to manually select the features used for classification. The application can extract 114 features
currently, which are shown in the table. The number in the bracket indicates the number of columns of features related to the 
features. 

For example: Mean (9) means 9 columns of means are extracted, so 3 for each of the sensors (i.e. accX, accY, accZ, rotX, rotY, 
rotZ, graX, graY, graZ).

Acerage Resultant Acceleration (3) has 3 columns, meaning one for each sensor.

|Features (number)|	Description|
| --- | --- |
|Mean (9)|	The average value of the data for each axis in the window|
|Standard Deviation (9)|	Standard deviation of each axis in the window|
|Variance (9)|	The square of the standard deviation of each axis in the window|
|Mean Absolute Deviation (9)|	The average difference between the mean and each of the values for each axis in the window|
|Minimum (9)|	The minimum value of the data for each axis in the window|
|Maximum (9)|	The maximum value of the data for each axis in the window|
|Inter-quartile Range (9)|	The range of the middle 50% of the values for each axis in the data|
|Average Resultant Acceleration (3)|	The average of the square roots of the sum of the squared value of 3 axis for each type of sensor in the data|
|Skewness (9)|	The degree of distortion of each axis from the symmetrical bell curve in the window|
|Kurtosis (9)|	The weight of the distribution tails for each axis in the window|
|Signal Magnitude Area (3)|	The normalized integral of 3-axis for each type of sensors in the window|
|Energy (9)|	The area under the squared magnitude of each axis in the window|
|Zero Crossing Rate (9)|	The number of times the data crossed the 0 value for each axis in the window|
|Number of Peaks (9)|	The number of peaks for each axis in the window|

We have used the correlation table to reduce the features from 114 to 71. The remaining features can be viewed in the code.
The feature reduction is done by removing all the features with absolute correlation coefficient greater than 0.9.
The feature reduction in code is done by commenting out the unwanted features or select only the wanted columns in the features.


**The feature extracted data and target would be saved in a folder called "TrainingSet", which would be used for classification**

**Two csv files would be generated, one is the data (features for the data), and the other is the target (the activity)**

### Classification Page
To perform classification, a smartphone must be paired with the machine first before this function can work.

Firstly, to perform any of the functions in the page, the user needs to select the training data and the target. Depending on 
the type of features used for classification (tsfresh or manual selection), the user need to toggle the button to match with
the training data (different methods generates different types of data). By clicking the "Classify" button, a window would be 
created to display the top three predictions the model produces (with the probability of the prediction).

7 different machine learning models have been tested with this application, and we currently discovered that the extremely
randomised tree algorithm (ExtraTree) is the best model for the classification.

#### Other functions
The other functions in the page includes generate correlation plot for the features, generate confusion matrix for the model, 
generate cross-validation result for the machine learning models, and performing a validation by randomise the target label to 
validate the reliability of the model.
