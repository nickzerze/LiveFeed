"# LiveFeed"

server.py is a Flask app on your server/laptop, using OpenCV to capture frames 
from the webcam, display them, and track a bounding box sent by the Android app.

OpenCV trackers:
1) BOOSTING: This algorithm is based on AdaBoost, a machine-learning method that boosts 
accuracy by focusing on misclassified instances. While reliable, it’s relatively slow 
and struggles with noisy environments.

2) MIL (Multiple Instance Learning): MIL uses a "bag" of positive and negative samples 
around the object to make predictions, making it more resistant to occlusion. 
However, it lacks the speed of more advanced algorithms.

3) KCF (Kernelized Correlation Filters): KCF is faster and more accurate than traditional 
methods, using correlation filters to predict an object’s next position. 
It’s ideal for scenarios with minimal scale changes but may struggle when the object is lost.

4) TLD (Tracking, Learning, Detection): This tracker combines detection and tracking 
for high robustness. It can recover from failures but has unpredictable behavior when handling 
fast or complex movements.

5) MedianFlow: A reliable tracker for slow and steady movements, MedianFlow excels 
at predicting failures but falters with abrupt motion changes.

6) MOSSE (Minimum Output Sum of Squared Error): MOSSE is fast and works well in high-frame-rate scenarios.
However, its accuracy is limited, making it suitable for simple, less dynamic environments.

7) CSRT (Channel and Spatial Reliability Tracking): One of the most accurate OpenCV trackers, 
CSRT performs well with rotation, scale, and occlusion variations. It’s slower, making it best suited 
for accuracy-critical applications.

8) GOTURN (Deep Learning-based): GOTURN utilizes a convolutional neural network for tracking, 
making it capable of handling complex scenarios. However, it requires a pre-trained model,
and its performance can vary based on environmental conditions.