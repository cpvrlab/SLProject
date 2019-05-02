Christoffelturm README
======================

Default Marker
--------------
The default Marker Image is the stones image which you can find under  _data/calibrations/vuforia_markers.pdf.

Customize Marker
----------------
The Marker Image can be changed by setting the SL_TRACKER_IMAGE_NAME flag in lib-SLProject.pro. There are currently three markers available:
    - Stones (found in _data/calibrations/vuforia_markers.pdf)
    - Road (found in _data/calibrations/vuforia_markers.pdf)
    - Abstract (found in _data/images/textures/abstract.png)
Generally any marker image can be used if it is placed in _data/images/textures/ as a png image.

Videodebug
----------
To enable the use of a video instead of a camera set the SL_VIDEO_DEBUG flag. This will change the behaviour of SLScene to load a video instead of a camera on launch. The path of the video ist currently defined to be _data/videos/testvid_+ $SL_TRACKER_IMAGE_NAME +.mp4. There is currently one testvideo for each of the three Markers. But others could be defined.

Debug Output
------------
To enable Debug Output set the SL_SAVE_DEBUG_OUTPUT flag. This will write multiple files into /tmp/cv_tracking/ on unix systems or into cv_tracking/ on Windows systems. It generates images on Keypoint detection, Matching, Reprojection and optical flow. This setting will impact performance heavily.

Flags in SLCVTrackerFeatures.h
------------------------------
- DEBUG_OUTPUT: This prints to stdout if a frame is Reposed or if it is tracked.
- FORCE_REPOSE: This will force the Repose and no frame will be tracked.
- DISTINGUISH_FEATURE_DETECT_COMPUTE: This will split the Detection and Description into two function call enabling better Timing information. This does impact performance and should usually be turned off.
- DRAW_*: These allow for certain options to be displayed on screen.

Customize Detector and Descriptor
---------------------------------
The Detector and Descriptor which are used can be switched over the menu Preferences->Video->(Detectors|Descriptors) currently these Settings are overwritten in the consturctor of SLCVTrackerFeatures if these Lines are removed the switching should work again. If DISTINGUISH_FEATURE_DETECT_COMPUTE is not set the descriptor is generally used to do everything. If it is set the Detector will be used for Detection and the Descriptor for description:
Not all Detectors and Descriptors can work together if this is the case the selection of the Descriptor will change your Detector.

Timing information
------------------
Under Infos->Statistics->Timing Statistics for Feature tracking will be displayed. Including the Times for Detection, Description and Matching.