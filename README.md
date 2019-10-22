# SmartphoneActivityClassifier
SmartphoneActivityClassifier is a python application that can collect real-time smartphone motion sensor data,
performing feature extraction and real-time classification for smartphone activities.

The data collection process is via bBluetooth from an Android app we developed https://github.com/LucasSherlock/ClassifySmartphoneActivity.

The data send from the Android app includes 11 columns of data, which includes 11 columns of data: the time recorded, 
3 axis signal from accelerometer, gyroscope and magnetometer, and the smartphone activity.

SmartphoneActivityClassifier can perform two different types of feature extraction:
- using existing library tsfresh to automatically perform feature extraction and reduction
- manually select features to extract.

