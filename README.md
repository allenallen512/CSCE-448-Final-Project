# CSCE-448-Final-Project

Tarun Arumugam, Allen Li, James Stautler

This project proposes to implement object removal from images. This involves not only removal of the object but also the filling of the hole created such that the new image is visually plausible. Such implementation will involve a combination of texture synthesis algorithms as well as inpainting techniques. We attempt to recreate the results of the paper “Region Filling and Object Removal by Exemplar-Based Image Inpainting”.

Dependencies:
1. pip install numpy open-cv python

Useful links:
1. https://github.com/bobqywei/inpainting-partial-conv/blob/master/inpaint.py#L106


Notes Region-Filling Algorithm: 
1. Select target region (omega) to be removed and filled. Phi (source) can be entire image minus the target region. (Phi = Image - Omega)
2. 
3. Template window (Psi) must be chosen (default = 9x9). should be slightly larger than the largest distinguishable element.

--Each pixel has a color value (empty if unfilled) and a confidence value -> once filled is frozen. Pixels along the fill front are given temporary priority value which determines the order they are filled. Priority computation is biased towards patches which are (i) continuation of strong edges and (ii) surrounded by high-confidence pixels
--

3. 
