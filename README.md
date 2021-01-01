# CannyEdgeDetection
Canny Edge Detection:

Multistage algorithm to detect wide range of edges in an image.
Developed by John. F Canny in 1986.

5 Steps:

1. Noise reduction   -> Gaussian Blur

2. Gradient Calculation   -> Compute the Edge intensity and direction  using
     image gradient e.g. use Sobel Operator

3. Non-Maxima Suppression    Check in the direction of the edge , if any of the 
     immediate pixel in the edge direction has high intensity then the current pixel,
      set the intensity if the current pixel to 0 otherwise keep the current pixels value..      
					
4. Double Threshold : Identify Strong, weak and non-relevant edges

5. Edge Tracking by Hysteresis: Convert the weak pixel to strong only if one of the 
Pixel around the processed pixel is strong otherwise set the pixel to 0.
