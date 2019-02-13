# Mars-Craters
Mars Craters Detection using Mask and Image Classification

### Key points :

**1. Use of U-Net algorithm** to get the mask of an image ie a segmented image of same size

<p float="left">
  <img src="img/mask-ex.png" width=600 /> 
</p>

Mask Prediction :

<p float="left">
  <img src="img/maskpred2.png" width=350 />
  <img src="img/maskpred.png" width=300 /> 
</p>

**2. Detection on the mask of circles with circular Hough transform**

<p float="left">
  <img src="img/maskpred_dots.png" width=600 /> 
</p>

However, using mask has some drawbacks : for instances, two many white area might lead to a presumed cicle :

<p float="left">
  <img src="img/maskpredlist.png" width=400 /> 
</p>

**Idea** : applying a classification (similarly used in sliding-windows method) in order to improve precision on detected circles. It will sort presumed circles obtained with the mask to add a confidence score.

**classification model** trained on smaller images, which can be either craters or random parts of the surface of mars.
<p float="left">
  <img src="img/smallcraters.png" width=600 /> 
</p>

**3. Combination of both mask and classification :**

<p float="left">
  <img src="img/maskpredlist2.png" width=600 /> 
</p>
