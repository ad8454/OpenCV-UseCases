# OpenCV-UseCases
This project showcases some practical application of OpenCV using its inbuilt modules.

**1. Image Enhancement:** For this problem, the RGB values of the original image are equalized between
0 and 255 to obtain a clearer picture.

<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/highway.png" width="400">
  <div align="center"><i><b>Original Image</b></i></div>
</p>
<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/highway_res1.png" width="400">
  <div align="center"><i><b>Enhanced Image</b></i></div>
</p>

**2. Wiener Deconvolution:** For this problem, the approach is to use Wiener deconvolution to stabilize the image. This can 
be done by testing the effect of deconvolution on the image with different kernels as PSF and varying values of NSR. 
However, this approach proved difficult for this image in OpenCV. Thus Matlab was used to deconvolve the image using 
Wiener deconvolution. This was done using the <i>deconvwnr</i> function with <i>NSR</i> = 0.003 and the unknown <i>PSF</i> 
as a Gaussian kernel of size 12 and standard deviation = 20. Legible results were obtained using these parameters but 
esults of similar quality could not be reproduced in OpenCV with the same parameters. 

However, on further trials, a 12x12 square kernel of all <i>1</i>s as the <i>PSF</i> and <i>NSR</i> = 0.002 was
found to yield acceptable results on the sample implementation of the Wiener deconvolution in OpenCV.


<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/carlicense_noisy.png" width="400">
  <div align="center"><i><b>Original Image</b></i></div>
</p>
<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res2_1.png" width="400">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res2_2.png" width="400">
  <div align="center"><i><b>Resored Image with Matlab(Left) and OpenCV(Right)</b></i></div>
</p>

**3. Hough Circles:** This problem was solved by utilizing the <i>HoughCircles</i> method of OpenCV which uses Hough 
transforms to identify circular shapes in the original image. The correctly identified circles are 
indicated by outlining them in blue, incorrect by red, and close to correct by orange. Thirty five 
out of the total thirty seven circles were correctly identified resulting in an accuracy of 94.59%

<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/cropcirlces.png" width="400">
  <div align="center"><i><b>Original Image</b></i></div>
</p>
<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res3.png" width="400">
  <div align="center"><i><b>Image with identified circles</b></i></div>
</p>


**5. Mean Shift Algorithm:** In this problem, the objective is to track the movement of the red ball as it moves
around in the video feed. For this, the user is first asked to trace a bounding box around the red ball in the
first frame of the video. The image inside the bounding box is then cropped and converted to HSV and its histogram
is computed. The back projection of this histogram and the coordinates of the bounding box are then provided as 
parameters for the <i>meanShift</i> function. This function returns the coordinates of the new location of the object 
of interest and this and its pack projection is again fed to the function for the next frame.

<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res5_1.png" width="300">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res5_2.png" width="300">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res5_3.png" width="300">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res5_4.png" width="300">
  <div align="center"><i><b>Video frames highlighting the tracked red ball</b></i></div>
</p>


**6. Image Segmentation:** For this problem, the original image is segmented into clusters using k-means 
clustering with k = 5. Each cluster is then repeatedly eroded and dilated to remove fragments and 
obtain a suitable segmented image of each berry type.

<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/berries.png" width="400">
  <div align="center"><i><b>Original Image</b></i></div>
</p>

| <img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res6_1.png" width="260"> | <img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res6_2.png" width="260"> | <img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/res6_3.png" width="260"> |
|---|---|---|
<p align="center">
  <div align="center"><i><b>Segmented images</b></i></div>
</p>



**7. Image Blending:** This problem was solved by computing the Gaussian pyramids of the original images 
upto level 5 and using them to compute the Laplacian pyramids. These were then successively merged and added 
up to obtain a blended version of the two images.

<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/apple.jpg" width="400">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/orange.jpg" width="400">
  <div align="center"><i><b>Original Images</b></i></div>
</p>
<p align="center">
	<img src="https://github.com/ad8454/OpenCV-UseCases/blob/master/res/resbon.png" width="400">
  <div align="center"><i><b>Blended Image</b></i></div>
</p>





