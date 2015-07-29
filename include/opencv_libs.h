//#############################################################################
//  File:      include/opencv_libs.h
//  Author:    Marcus Hudritsch
//  Date:      July 2014
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifdef _DEBUG
#pragma comment(lib,"ippicvmt.lib")
#pragma comment(lib,"opencv_stitching300d.lib")
#pragma comment(lib,"opencv_videostab300d.lib")
#pragma comment(lib,"opencv_shape300d.lib")
#pragma comment(lib,"opencv_objdetect300d.lib")
#pragma comment(lib,"opencv_photo300d.lib")
#pragma comment(lib,"opencv_calib3d300d.lib")
#pragma comment(lib,"opencv_features2d300d.lib")
#pragma comment(lib,"opencv_flann300d.lib")
#pragma comment(lib,"opencv_highgui300d.lib")
#pragma comment(lib,"opencv_videoio300d.lib")
#pragma comment(lib,"opencv_imgcodecs300d.lib")
#pragma comment(lib,"opencv_ml300d.lib")
#pragma comment(lib,"opencv_video300d.lib")
#pragma comment(lib,"opencv_imgproc300d.lib")
#pragma comment(lib,"opencv_core300d.lib")
#pragma comment(lib,"opencv_hal300d.lib")
#else
#pragma comment(lib,"ippicvmt.lib")
#pragma comment(lib,"opencv_stitching300.lib")
#pragma comment(lib,"opencv_videostab300.lib")
#pragma comment(lib,"opencv_shape300.lib")
#pragma comment(lib,"opencv_objdetect300.lib")
#pragma comment(lib,"opencv_photo300.lib")
#pragma comment(lib,"opencv_calib3d300.lib")
#pragma comment(lib,"opencv_features2d300.lib")
#pragma comment(lib,"opencv_flann300.lib")
#pragma comment(lib,"opencv_highgui300.lib")
#pragma comment(lib,"opencv_videoio300.lib")
#pragma comment(lib,"opencv_imgcodecs300.lib")
#pragma comment(lib,"opencv_ml300.lib")
#pragma comment(lib,"opencv_video300.lib")
#pragma comment(lib,"opencv_imgproc300.lib")
#pragma comment(lib,"opencv_core300.lib")
#pragma comment(lib,"opencv_hal300.lib")
#endif