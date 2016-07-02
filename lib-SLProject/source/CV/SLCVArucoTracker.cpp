//#############################################################################
//  File:      SLCVArucoTracker.cpp
//  Author:    Michael Göttlicher & Marcus Hudritsch
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLSceneView.h>
#include <SLCVArucoTracker.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
bool SLCVArucoTracker::init(string paramsFileDir)
{
    return _params.loadFromFile(paramsFileDir);
}
//-----------------------------------------------------------------------------
bool SLCVArucoTracker::track(cv::Mat image, 
                             SLCVCalibration& calib,
                             SLSceneView* sv)
{
    if(image.empty() || 
       calib.intrinsics().empty() || 
       _params.arucoParams.empty() || 
       _params.dictionary.empty() ||
       _node == nullptr)
       return false;
   
    vector<vector<Point2f>> corners, rejected;
    vector<Vec3d> rVecs, tVecs;

    //clear detected oject view matrices from last frame
    _arucoIDs.clear();
    _objectViewMats.clear();

    aruco::detectMarkers(image, 
                         _params.dictionary, 
                         corners, 
                         _arucoIDs, 
                         _params.arucoParams, 
                         rejected);

    if(_arucoIDs.size() > 0)
    {
        cout << "Aruco IdS: " << _arucoIDs.size() << " : ";

        aruco::estimatePoseSingleMarkers(corners, 
                                         _params.edgeLength, 
                                         calib.intrinsics(), 
                                         calib.distortion(), 
                                         rVecs,
                                         tVecs);

        for(size_t i=0; i < _arucoIDs.size(); ++i)
        {
            cout << _arucoIDs[i] << ",";

            // Convert cv translation & rotation vector to OpenGL matrix
            SLMat4f ovm = calib.createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));

            _objectViewMats.push_back(ovm);
        }
        cout << endl;

        //calculate object transformation matrix (see also calcObjectMatrix)
        SLMat4f om = sv->camera()->om() * _objectViewMats[0];
        
        _node->om(om);
        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
    }
    else
        _node->setDrawBitsRec(SL_DB_HIDDEN, true);

    return true;
}
//-----------------------------------------------------------------------------
/*! Explains the calculation of the object matrix from the cameraObject and the
object view matrix in detail:

Nomenclature:
T = homogenious transformation matrix

a
 T = homogenious transformation matrix with subscript b and superscript a
  b

Subscrips and superscripts:  w = world  o = object  c = camera

c
 T  = Transformation of object with respect to camera coordinate system.
  o   It discribes the position of an object in the camera coordinate system.
We get this Transformation from openCVs solvePNP function.

w       c    -1
 T  = ( T )    = Transformation of camera with respect to world coord.-system.
  c       w        Inversion exchanges sub- and superscript.
This is also called the view matrix.

We can combine two or more homogenious transformations to a new one if the
inner sub- and superscript fit together. The resulting transformation
inherits the superscrip from the left and the subscript from the right
transformation. The following tranformation is what we want to do:

w    w     c
 T  = T  *  T   = Transformation of object with respect to world
  o    c     o    coordinate system (object matrix)
*/
static void calcObjectMatrix(const SLMat4f& cameraObjectMat, 
                             const SLMat4f& objectViewMat, 
                             SLMat4f& objectMat)
{   
    // new object matrix = camera object matrix * object-view matrix
    objectMat = cameraObjectMat * objectViewMat;
}
//-----------------------------------------------------------------------------
void SLCVArucoTracker::drawArucoMarkerBoard(int dictionaryId,
                                            int numMarkersX,
                                            int numMarkersY, 
                                            int markerEdgePX, 
                                            int markerSepaPX,
                                            string imgName, 
                                            bool showImage, 
                                            int borderBits, 
                                            int marginsSize)
{
    if(marginsSize == 0)
        marginsSize = markerSepaPX;

    Size imageSize;
    imageSize.width  = numMarkersX * (markerEdgePX + markerSepaPX) - markerSepaPX + 2 * marginsSize;
    imageSize.height = numMarkersY * (markerEdgePX + markerSepaPX) - markerSepaPX + 2 * marginsSize;

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX, 
                                                           numMarkersY, 
                                                           float(markerEdgePX),
                                                           float(markerSepaPX), 
                                                           dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, marginsSize, borderBits);

    if(showImage) 
    {   imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(imgName, boardImage);
}
//-----------------------------------------------------------------------------
void SLCVArucoTracker::drawArucoMarker(int dictionaryId,
                                       int minMarkerId,
                                       int maxMarkerId,
                                       int markerSizePX)
{
    assert(dictionaryId > 0);
    assert(minMarkerId > 0);
    assert(minMarkerId < maxMarkerId);

    using namespace aruco;

    Ptr<Dictionary> dict = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(dictionaryId));
    if (maxMarkerId > dict->bytesList.rows)
        maxMarkerId = dict->bytesList.rows;

    Mat markerImg;

    for (int i=minMarkerId; i<maxMarkerId; ++i)
    {   drawMarker(dict, i, markerSizePX, markerImg, 1);
        char name[255];
        sprintf(name, 
                "ArucoMarker_Dict%d_%dpx_Id%d.png", 
                dictionaryId, 
                markerSizePX, 
                i);

        imwrite(name, markerImg);
    }
}
//-----------------------------------------------------------------------------