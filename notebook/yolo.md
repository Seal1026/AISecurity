#### Training dataset
- Anchor box for label in training set: The vector that describes detected object iformation (object's center lie in the grid box). Once grid box will have one anchor bos. For an object, the output will be a column vector with length = 1(for whether there is object) + 2(coordinates of center) + 2(width and height) + c (number of classes). <br/>

First write the image as a i x i grid. For each small region, we have a 1xl vector. The training data x_train will be images. Therefore, for an image label y_train in a training set, it will be a vector with size ixixl.<br/>
#### Model
Convolutional neural network
![image](https://user-images.githubusercontent.com/50323850/133725068-a357b366-7f28-4202-9dd1-855354a968dd.png)
#### Problem with prediction
- There might be multiple boundingbox in the same object (same class). <br/>
  - Solution: Cannot take maximum probability score in each calss as there might be multiple objects in the same class; otherwise some correct bounding box might be discarded. Therefore, we use IOU(intersection over union) to discard the overlapping bonding box.
  - IOU is intersection area/union area.  We set a certain threshold lets say 0.7. If the IOU of the two bounding box in the same class larger than this value we will leave the box with higher probability ad discard the one with lower probability.
![image](https://user-images.githubusercontent.com/50323850/133725670-de658843-fa3a-45a7-b9cc-38a2bd5e6f28.png)

#### Problem with traingset label
- What if there is two anchor box for one grid? How to label it using anchor box?
   - Solution: combine it into a larger column vector=> size of anchor box: 1x(numberxsizeforone).


