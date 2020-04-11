import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import accuracy_test
from accuracy_test import accuracy
import numpy as np
import matplotlib.pyplot as plt
from vggface_model.vggface_utils import VggfaceUtils
from vggface_model.face_detection import DetectionUtils

'''
This is a module used for doing optimizing the threshold in the recognition function.
We go through 50 to 100 and test the recognition accuracy.
Find the best threshold by getting the highest accuracy.
   
'''

def main():
    vggfaceUtils = VggfaceUtils()
    detectionUtils = DetectionUtils()
    x = np.arange(50,150,5)
    accu = []
    for i in x:
        accu.append(accuracy(vggfaceUtils, detectionUtils, i))
        print(accu)
    best_accurate_rate = max(accu)
    Max_index = accu.index(best_accurate_rate)
    best_threshold = x[Max_index]
    
    print("The best threshold is {}, and the accurate rate is {}:".format(best_threshold,best_accurate_rate))
    plt.plot(x,accu)
    plt.show()
if __name__ == "__main__":
    main()