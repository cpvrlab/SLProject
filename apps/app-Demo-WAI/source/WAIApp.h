//#############################################################################
//  File:      WAISceneView.h
//  Purpose:   Node transform test application that demonstrates all transform
//             possibilities of SLNode
//  Author:    Marc Wacker
//  Date:      July 2015
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef APP_WAI_SCENE_VIEW
#define APP_WAI_SCENE_VIEW

#include <vector>
#include <functional>
#include "AppWAIScene.h"

#include <CVCalibration.h>
#include <WAIAutoCalibration.h>
#include <DeviceData.h>
#include <AppDemoWaiGui.h>
#include <SLInputEventInterface.h>
#include <WAISlam.h>
#include <SENSCamera.h>
#include <SENSVideoStream.h>
#include <GLSLextractor.h>
#include <FeatureExtractorFactory.h>
#include <SLInputManager.h>
#include <WAIEvent.h>
#include <SlamParams.h>

#include <states/SelectionState.h>
#include <states/StartUpState.h>
#include <states/AreaTrackingState.h>
#include <states/CameraTestState.h>
#include <states/LocationMapState.h>
#include <states/TestState.h>
#include <states/TutorialState.h>

class SLMaterial;
class SLPoints;
class SLNode;
class AppDemoGuiError;
class AppDemoGuiSlamLoad;

class WAIApp : public SLInputEventInterface
{
public:
    using CloseAppCallback = std::function<void(void)>;
    enum class State
    {
        /*!Wait for _startFromIdle to become true. When it is true start SelectionState for scene selection and when it is
        started switch to START_UP state or directly start StartUpScene if AppMode is already not AppMode::NONE.
        */
        IDLE,
        /*!In this state the user has to select a an AppMode. When selection is not NONE, we switch to state START_UP
        */
        SELECTION,
        /*!We start up the states depending on selected AppMode
        */
        START_UP,
        LOCATION_MAP,
        AREA_TRACKING,
        TEST,
        CAMERA_TEST,
        TUTORIAL
    };

    WAIApp();
    ~WAIApp();
    //call load to correctly initialize wai app
    int  load(int scrWidth, int scrHeight, float scr2fbX, float scr2fbY, int dpi, AppDirectories directories);
    void setCamera(SENSCamera* camera);
    //call update to update the frame, wai and visualization
    bool update();
    void close();
    void terminate();

    std::string name();

    //! set a callback function which can be used to inform caller that app wants to be closed
    void setCloseAppCallback(CloseAppCallback cb) { _closeCB = cb; }
    //! caller informs that app back button was pressed
    void goBack()
    {
        _goBack = true;
    }

private:
    SENSCamera* getCamera();

    void reset();
    void checkStateTransition();
    bool updateState();

    AppDirectories _dirs;

    SENSCamera* _camera = nullptr;
    std::mutex  _cameraSetMutex;

    std::string    _name;
    SLInputManager _inputManager;

    std::unique_ptr<DeviceData> _deviceData;

    State              _state             = State::IDLE;
    SelectionState*    _selectionState    = nullptr;
    StartUpState*      _startUpState      = nullptr;
    AreaTrackingState* _areaTrackingState = nullptr;
    CameraTestState*   _cameraTestState   = nullptr;
    LocationMapState*  _locationMapState  = nullptr;
    TestState*         _testState         = nullptr;
    TutorialState*     _tutorialState     = nullptr;

    //Sub-States that lead to state start() call:
    //set to true as soon as we have access to app resouces, then we can first visualize something
    bool _startFromIdle = false;
    //done after AppMode was selected, then we know what to start up
    bool _startFromStartUp = false;

    //defines if ErlebAR scene was already selected or if user has to choose
    AppMode _appMode            = AppMode::NONE;
    Area    _area               = Area::NONE;
    bool    _showSelectionState = false;

    bool _switchToTracking = false;

    CloseAppCallback _closeCB = nullptr;
    bool             _goBack  = false;
};

#endif
