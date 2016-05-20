//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include "ARArucoTracker.h"
#include <ARSceneView.h>
#include <SLBox.h>
#include <SLAssimpImporter.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

//-----------------------------------------------------------------------------
ARArucoTracker::ARArucoTracker(Mat intrinsics, Mat distoriton) :
    ARTracker(intrinsics, distoriton)
{
}
//-----------------------------------------------------------------------------
bool ARArucoTracker::init(string paramsFileDir)
{
    return _p.loadFromFile(paramsFileDir);
}
//-----------------------------------------------------------------------------
bool ARArucoTracker::track()
{
    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    vector< Vec3d > rvecs, tvecs;
    //clear detected Objects from last frame
    _arucoVMs.clear();

    if(!_image.empty() && !_intrinsics.empty() && !_p.arucoParams.empty() && !_p.dictionary.empty())
    {
        aruco::detectMarkers(_image, _p.dictionary, corners, ids, _p.arucoParams, rejected);

        if( ids.size() > 0)
        {
            aruco::estimatePoseSingleMarkers(corners, _p.edgeLength, _intrinsics, _distortion, rvecs,
                                                         tvecs);

            for( size_t i=0; i < rvecs.size(); ++i)
            {
                //Transform calculated position (rotation and translation vector) from openCV to SLProject form
                //as discribed in this post:
                //http://www.morethantechnical.com/2015/02/17/augmented-reality-on-libqglviewer-and-opencv-opengl-tips-wcode/
                //attention: We dont have to transpose the resulting matrix, because SLProject uses row-major matrices.
                //For direct openGL use you have to transpose the resulting matrix additionally.

                //convert vector to rotation matrix
                Rodrigues(rvecs[i], _rMat);
                _tVec = Mat(tvecs[i]);

                //convert to SLMat4f:
                //y- and z- axis have to be inverted
                /*
                      |  r00   r01   r02   t0 |
                      | -r10  -r11  -r12  -t1 |
                  m = | -r20  -r21  -r22  -t2 |
                      |    0     0     0    1 |
                */

                SLMat4f vm;
                //1st row
                vm(0,0) = _rMat.at<double>(0,0);
                vm(0,1) = _rMat.at<double>(0,1);
                vm(0,2) = _rMat.at<double>(0,2);
                vm(0,3) = _tVec.at<double>(0,0);
                //2nd row
                vm(1,0) = -_rMat.at<double>(1,0);
                vm(1,1) = -_rMat.at<double>(1,1);
                vm(1,2) = -_rMat.at<double>(1,2);
                vm(1,3) = -_tVec.at<double>(1,0);
                //3rd row
                vm(2,0) = -_rMat.at<double>(2,0);
                vm(2,1) = -_rMat.at<double>(2,1);
                vm(2,2) = -_rMat.at<double>(2,2);
                vm(2,3) = -_tVec.at<double>(2,0);
                //4th row
                vm(3,0) = 0.0f;
                vm(3,1) = 0.0f;
                vm(3,2) = 0.0f;
                vm(3,3) = 1.0f;

                _arucoVMs.insert( pair<int,SLMat4f>(ids[i], vm));
            }
        }
        return true;
    }

    return false;
}
//-----------------------------------------------------------------------------
static void calcObjectMatrix(const SLMat4f& cameraObjectMat, SLMat4f& objectViewMat, SLMat4f& objectMat )
{
    //calculate objectmatrix:
    /*
     * Nomenclature:
     * T = homogenious transformation matrix
     *
     * a
     *  T = homogenious transformation matrix with subscript b and superscript a
     *   b
     *
     * Subscrips and superscripts:
     * w = world
     * o = object
     * c = camera
     *
     * c
     *  T  = Transformation of object with respect to camera coordinate system. It discribes
     *   o   the position of an object in the camera coordinate system. We get this Transformation
     *       from openCVs solvePNP function.
     *
     * w       c    -1
     *  T  = (  T  )    = Transformation of camera with respect to world coordinate system. Inversion exchanges
     *   c       w        sub- and superscript
     *
     * We can combine two or more homogenious transformations to a new one if the inner sub- and superscript
     * fit together. The resulting transformation inherits the superscrip from the left and the subscript from
     * the right transformation.
     * The following tranformation is what we want to do:
     *  w    w     c
     *   T  = T  *  T   = Transformation of object with respect to world coordinate system (object matrix)
     *    o    c     o
     */

    //new object matrix = camera object matrix * object-view matrix
    objectMat = cameraObjectMat * objectViewMat;
}
//-----------------------------------------------------------------------------
void ARArucoTracker::updateSceneView( ARSceneView* sv )
{
    //container for updated nodes
    std::map<int,SLNode*> updatedNodes;

    //get new ids and transformations from tracker
    //std::map<int,SLMat4f>& vms = getArucoVMs();

    bool newNodesAdded = false;
    //for all new items
    for( const auto& pair : _arucoVMs)
    {
        int key = pair.first;
        SLMat4f ovm = pair.second;
        //calculate object transformation matrix
        SLMat4f om;
        calcObjectMatrix(sv->camera()->om(), ovm, om );

        //check if there is already an object for this id in existingObjects
        auto it = _arucoNodes.find(key);
        if( it != _arucoNodes.end()) //if object already exists
        {
            //set object transformation matrix
            SLNode* node = it->second;
            //set new object matrix
            node->om( om );
            //set unhidden
            node->setDrawBitsRec( SL_DB_HIDDEN, false );

            //add to container newObjects
            updatedNodes.insert( /*std::pair<int,SLNode*>(key, node)*/ *it );
            //remove it from container existing objects
            _arucoNodes.erase(it);
        }
        else //object does not exist
        {
            //create a new object
            float r = ( rand() % 10 + 1 ) / 10.0f;
            float g = ( rand() % 10 + 1 ) / 10.0f;
            float b = ( rand() % 10 + 1 ) / 10.0f;
            SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(r,g,b));
            stringstream ss; ss << "Box" << key;
            SLNode* box = new SLNode( ss.str() );

            box->addMesh(new SLBox(-_p.edgeLength/2, -_p.edgeLength/2, 0.0f, _p.edgeLength/2, _p.edgeLength/2, _p.edgeLength, "Box", rMat));
            //set object transformation matrix
            box->om( om );

            // load coordinate axis arrows
            SLAssimpImporter importer;
            SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
            axesNode->scale(0.3);
            box->addChild(axesNode);

            //add new object to Scene
            SLScene::current->root3D()->addChild(box);
            newNodesAdded = true;

            //add to container of updated objects
            updatedNodes.insert( std::pair<int,SLNode*>(key, box));

            cout << "Aruco Markers: Added new object for id " << key << endl << endl;
        }
    }

    //for all remaining objects in existingObjects
    for( auto& it : _arucoNodes )
    {
        //hide
        SLNode* node = it.second;
        node->setDrawBitsRec( SL_DB_HIDDEN, true );
        //add to updated objects
        updatedNodes.insert( it );
    }

    //update aabbs if new nodes added
    if( newNodesAdded )
        SLScene::current->root3D()->updateAABBRec();

    //overwrite aruco nodes
    _arucoNodes = updatedNodes;
}
//-----------------------------------------------------------------------------
void ARArucoTracker::unloadSGObjects()
{
    for( auto& it : _arucoNodes )
    {
        SLNode* node = it.second;
        SLNode* parent = node->parent();
        parent->deleteChild(node);
        node = nullptr;
        parent->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void ARArucoTracker::drawArucoMarkerBoard(int numMarkersX, int numMarkersY, int markerEdgeLengthPix, int markerSepaPix,
                                     int dictionaryId, string imgName, bool showImage, int borderBits, int marginsSize )
{
    if(marginsSize == 0)
        marginsSize = markerSepaPix;

    Size imageSize;
    imageSize.width = numMarkersX * (markerEdgeLengthPix + markerSepaPix) - markerSepaPix + 2 * marginsSize;
    imageSize.height = numMarkersY * (markerEdgeLengthPix + markerSepaPix) - markerSepaPix + 2 * marginsSize;

    Ptr<aruco::Dictionary> dictionary =
     aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(numMarkersX, numMarkersY, float(markerEdgeLengthPix),
                                                   float(markerSepaPix), dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, marginsSize, borderBits);

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(imgName, boardImage);
}
//-----------------------------------------------------------------------------
