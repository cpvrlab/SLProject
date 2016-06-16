//#############################################################################
//  File:      ARSceneView.h
//  Purpose:   Augmented Reality Demo
//  Author:    Michael G�ttlicher
//  Date:      Spring 2016
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael G�ttlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <SLSceneView.h>
#include <ARTracker.h>
#include <SLCVCalibration.h>
#include <AR2DMapper.h>

//-----------------------------------------------------------------------------

/*!
SLSceneView derived class for a node transform test application that
demonstrates all transform possibilities in SLNode
*/
class ARSceneView : public SLSceneView
{
    public:
        enum ARSceneViewMode
        {   Idle,
            CalibrationMode,
            ChessboardMode,
            ArucoMode,
            Mapper2D,
            Tracker2D
        };

                        ARSceneView         (string calibFileDir,
                                             string paramFilesDir);
                       ~ARSceneView         ();

        ARTracker*      tracker             () {return _tracker;}

        // From SLSceneView overwritten
        void            preDraw             ();
        void            postDraw            ();
        void            postSceneLoad       ();

        SLbool          onKeyPress          (const SLKey key, const SLKey mod);

        SLCVCalibration&  calibration         ()  { return _calibMgr; }
        void            clearInfoLine       ();
        void            setInfoLineText     (SLstring text);

private:
        //bool            setTextureToCVImage (cv::Mat& image);
        void            setCVImageToTexture (cv::Mat& image);

        void            renderText          ();
        void            updateInfoText      ();

        bool            loadCamParams       (string filename);
        void            calcCameraFOV       ();
        void            processModeChange   ();

        //std::map<int,SLNode*> _arucoNodes;
        ARTracker*      _tracker;            //!< Tracker instance

        SLText*         _infoText;           //!< node for all text display
        SLText*         _infoBottomText;     //!< node for all text display
        SLstring        _infoLine;

        ARSceneViewMode _newMode;
        ARSceneViewMode _currMode;

        string          _calibFileDir;      //!< path to calibration files are stored
        string          _paramFilesDir;     //!< path to parameter files are stored

        SLCVCalibration   _calibMgr;          //!< calibration manager
        AR2DMapper      _mapper2D;          //!< 2D Mapping
};
//-----------------------------------------------------------------------------