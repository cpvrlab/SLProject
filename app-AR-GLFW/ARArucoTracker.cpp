//#############################################################################
//  File:      ARTracker.cpp
//  Author:    Michael Göttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Göttlicher
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
    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;
    vector<Vec3d> rvecs, tvecs;
    
    //clear detected oject view matrices from last frame
    _arucoOVMs.clear(); 

    if(!_image.empty() && !_intrinsics.empty() && !_p.arucoParams.empty() && !_p.dictionary.empty())
    {
        aruco::detectMarkers(_image, _p.dictionary, corners, ids, _p.arucoParams, rejected);

        if(ids.size() > 0)
        {
            cout << "Aruco IdS: " << ids.size() << " : ";

            aruco::estimatePoseSingleMarkers(corners, 
                                             _p.edgeLength, 
                                             _intrinsics, 
                                             _distortion, 
                                             rvecs,
                                             tvecs);

            for(size_t i=0; i < rvecs.size(); ++i)
            {
                cout << ids[i] << ",";

                // Convert vector to rotation matrix
                Rodrigues(rvecs[i], _rMat);
                _tVec = Mat(tvecs[i]);

                // Convert cv translation & rotation to OpenGL transform matrix
                SLMat4f ovm = cvMatToGLMat(_tVec, _rMat);

                _arucoOVMs.insert(pair<int,SLMat4f>(ids[i], ovm));
            }
            cout << endl;
        }
        return true;
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
void ARArucoTracker::updateSceneView(ARSceneView* sv)
{
    //container for updated nodes
    std::map<int,SLNode*> updatedNodes;

    bool newNodesAdded = false;

    //for all new items
    for(const auto& pair : _arucoOVMs)
    {
        int key = pair.first;
        SLMat4f ovm = pair.second;

        //calculate object transformation matrix (see also calcObjectMatrix)
        SLMat4f om = sv->camera()->om() * ovm;

        //check if there is already an object for this id in existingObjects
        auto it = _arucoNodes.find(key);
        if(it != _arucoNodes.end()) //if object already exists
        {
            //set object transformation matrix
            SLNode* node = it->second;

            //set new object matrix
            node->om(om);

            //set unhidden
            node->setDrawBitsRec(SL_DB_HIDDEN, false);

            //add to container newObjects
            updatedNodes.insert(/*std::pair<int,SLNode*>(key, node)*/ *it);

            //remove it from container existing objects
            _arucoNodes.erase(it);
        }
        else //object does not exist
        {
            //create a new object
            float r = (rand() % 10 + 1) / 10.0f;
            float g = (rand() % 10 + 1) / 10.0f;
            float b = (rand() % 10 + 1) / 10.0f;
            SLMaterial* rMat = new SLMaterial("rMat", SLCol4f(r,g,b));
            stringstream ss; ss << "Box" << key;
            SLNode* box = new SLNode(ss.str());

            box->addMesh(new SLBox(-_p.edgeLength/2, -_p.edgeLength/2, 0.0f, _p.edgeLength/2, _p.edgeLength/2, _p.edgeLength, "Box", rMat));
            //set object transformation matrix
            box->om(om);

            // load coordinate axis arrows
            SLAssimpImporter importer;
            SLNode* axesNode = importer.load("FBX/Axes/axes_blender.fbx");
            axesNode->scale(0.3f);
            box->addChild(axesNode);

            //add new object to Scene
            SLScene::current->root3D()->addChild(box);
            newNodesAdded = true;

            //add to container of updated objects
            updatedNodes.insert(std::pair<int,SLNode*>(key, box));
        }
    }

    //for all remaining objects in existingObjects
    for(auto& it : _arucoNodes)
    {
        //hide
        SLNode* node = it.second;
        node->setDrawBitsRec(SL_DB_HIDDEN, true);

        //add to updated objects
        updatedNodes.insert(it);
    }

    //update aabbs if new nodes added
    if(newNodesAdded)
        SLScene::current->root3D()->updateAABBRec();

    //overwrite aruco nodes
    _arucoNodes = updatedNodes;
}
//-----------------------------------------------------------------------------
void ARArucoTracker::unloadSGObjects()
{
    for(auto& it : _arucoNodes)
    {
        SLNode* node = it.second;
        SLNode* parent = node->parent();
        parent->deleteChild(node);
        node = nullptr;
        parent->updateAABBRec();
    }
}
//-----------------------------------------------------------------------------
void ARArucoTracker::drawArucoMarkerBoard(int dictionaryId,
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
void ARArucoTracker::drawArucoMarker(int dictionaryId,
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