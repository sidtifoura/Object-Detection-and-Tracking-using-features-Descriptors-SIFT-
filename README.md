# Object-Detection-and-Tracking-using-features-Descriptors-SIFT-
## Feature Extraction

We describe an object of interest using the Scale-Invariant Feature Transform (SIFT) algorithm. SIFT is designed to identify distinctive keypoints in an image that are both scale-invariant and rotation-invariant. These keypoints play a crucial role in ensuring accurate tracking of the object across multiple frames, even when its appearance undergoes changes. It is essential to identify keypoints that remain consistent regardless of variations in viewing distance or angle, thereby providing scale and rotation invariance.

## Feature Matching

For feature matching, we employ the Fast Library for Approximate Nearest Neighbors (FLANN). This step involves establishing correspondences between keypoints to determine whether a frame contains keypoints similar to those associated with our object of interest. Upon finding a good match, we can effectively mark the object in each frame.

## Feature Tracking

To track the located object of interest from frame to frame, we implement various forms of early outlier detection and rejection. This enhances the algorithm's efficiency by swiftly handling discrepancies and ensuring robust tracking performance.

## Perspective Transformation

To compensate for any translations and rotations undergone by the object, we apply a perspective transform. This process warps the perspective, aligning the object to appear upright at the center of the screen. This transformation creates a captivating visual effect where the object appears frozen in a fixed position while the surrounding scene rotates dynamically around it.
