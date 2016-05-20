//#############################################################################
//  File:      ARSceneView.h
//  Purpose:   Augmented Reality Demo
//  Author:    Michael Göttlicher
//  Date:      May 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <ARTracker.h>

//-----------------------------------------------------------------------------

/*!
SLSceneView derived class for a node transform test application that
demonstrates all transform possibilities in SLNode
*/
class ARSceneView : public SLSceneView
{
    public:
        enum ARSceneViewMode
        {
            CalibrationMode,
            Idle,
            ChessboardMode,
            ArucoMode
        };

        ARSceneView( string calibFileDir, string paramFilesDir);
        ~ARSceneView();

        ARTracker*      tracker         () {return _tracker;}

        // From SLSceneView overwritten
        void            preDraw         ();
        void            postDraw        ();
        void            postSceneLoad   ();

        SLbool          onKeyPress      (const SLKey key, const SLKey mod);

        float           getCameraFov    () { return _cameraFovDeg; }

    private:
        void            loadNewFrameIntoTracker();

        void            renderText      ();
        void            updateInfoText  ();

        void            loadCamParams   (string filename);
        void            calculateCameraFieldOfView();

//        std::map<int,SLNode*> _arucoNodes;
        ARTracker*      _tracker;            //!< Tracker instance

        SLText*         _infoText;      //!< node for all text display

        ARSceneViewMode _newMode;
        ARSceneViewMode _currMode;

        //camera intrinsic parameter
        cv::Mat         _intrinsics;
        //camera distortion parameter
        cv::Mat         _distortion;
        // camera field of view
        float           _cameraFovDeg;

        //directory, where the calibration files are stored
        string          _calibFileDir;
        //directory, where the parameter files are stored
        string          _paramFilesDir;
};
//-----------------------------------------------------------------------------
