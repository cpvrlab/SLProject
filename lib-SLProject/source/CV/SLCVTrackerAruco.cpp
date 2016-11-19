//#############################################################################
//  File:      SLCVTrackerAruco.cpp
//  Author:    Michael Göttlicher & Marcus Hudritsch
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>         // precompiled headers
#include <SLSceneView.h>
#include <SLCVTrackerAruco.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
//-----------------------------------------------------------------------------
// Initialize static variables
bool            SLCVTrackerAruco::trackAllOnce = true;
bool            SLCVTrackerAruco::paramsLoaded = false;
SLVint          SLCVTrackerAruco::arucoIDs;
SLVMat4f        SLCVTrackerAruco::objectViewMats;
SLCVArucoParams SLCVTrackerAruco::params;
//-----------------------------------------------------------------------------
SLCVTrackerAruco::SLCVTrackerAruco(SLNode* node, SLint arucoID) : 
                  SLCVTracker(node) 
{
    _arucoID = arucoID;
}
//-----------------------------------------------------------------------------
bool SLCVTrackerAruco::track(cv::Mat image, 
                             SLCVCalibration& calib,
                             SLVSceneView& sceneViews)
{
    assert(!image.empty() && "Image is empty");
    assert(!calib.intrinsics().empty() && "Calibration is empty");
    assert(_node && "Node pointer is null");
   
    // Load aruco parameter once
    if (!paramsLoaded)
    {   paramsLoaded = params.loadFromFile();
        if (!paramsLoaded)
            SL_EXIT_MSG("SLCVTrackerAruco::track: Failed to load Aruco parameters.");
    }
    if(params.arucoParams.empty() || params.dictionary.empty())
    {   SL_WARN_MSG("SLCVTrackerAruco::track: Aruco paramters are empty.");
        return false;
    }

    // Track all Aruco markers only once per frame
    if (trackAllOnce)
    {   arucoIDs.clear();
        objectViewMats.clear();
        vector<vector<cv::Point2f>> corners, rejected;

        aruco::detectMarkers(image, 
                             params.dictionary, 
                             corners, 
                             arucoIDs, 
                             params.arucoParams, 
                             rejected);

        if(arucoIDs.size() > 0)
        {   cout << "Aruco IdS: " << arucoIDs.size() << " : ";
            vector<cv::Vec3d> rVecs, tVecs;

            aruco::estimatePoseSingleMarkers(corners, 
                                             params.edgeLength, 
                                             calib.intrinsics(), 
                                             calib.distortion(), 
                                             rVecs, tVecs);

            for(size_t i=0; i < arucoIDs.size(); ++i)
            {   cout << arucoIDs[i] << ",";
                SLMat4f ovm = calib.createGLMatrix(cv::Mat(tVecs[i]), cv::Mat(rVecs[i]));
                objectViewMats.push_back(ovm);
            }
            cout << endl;
        }
        trackAllOnce = false;
    }

    if(arucoIDs.size() > 0)
    {   
        // Find the marker with the matching id
        for(size_t i=0; i < arucoIDs.size(); ++i)
        {   if (arucoIDs[i] == _arucoID)
            {   
                for (auto sv : sceneViews)
                {
                    if (_node == sv->camera())
                        _node->om(objectViewMats[i].inverse());
                    else
                    {   //calculate object transformation matrix (see also calcObjectMatrix)
                        _node->om(sv->camera()->om() * objectViewMats[i]);
                        _node->setDrawBitsRec(SL_DB_HIDDEN, false);
                    }
                }
            }
        }
        return true;
    } else
    {
        // Hide tracked node if not visible
        for (auto sv : sceneViews)
            if (_node != sv->camera())
                _node->setDrawBitsRec(SL_DB_HIDDEN, true);
    }

    return false;
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
void SLCVTrackerAruco::drawArucoMarkerBoard(int dictionaryId,
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
void SLCVTrackerAruco::drawArucoMarker(int dictionaryId,
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
