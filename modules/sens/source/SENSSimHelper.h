#ifndef SENS_SIMHELPER_H
#define SENS_SIMHELPER_H

#include <SENSSimulator.h>
#include <SENSRecorder.h>

class SENSSimHelper
{
public:
    //The camera pointer has to be used as reference
    SENSSimHelper(SENSGps*&                 gps,
                  SENSOrientation*&         orientation,
                  SENSCamera*&              camera,
                  const std::string&        simDataDir,
                  std::function<void(void)> cameraParametersChangedCB);
    ~SENSSimHelper();

    bool startRecording();
    void stopRecording();
    void updateGpsRecording();
    void updateOrientationRecording();
    void updateCameraRecording();
    bool recorderIsRunning();
    void resetRecorder();
    bool getRecorderErrors(std::vector<std::string>& errorMsgs);

    bool canSimGps();
    bool canSimOrientation();
    bool canSimCamera();

    void initSimulator(const std::string& simDataSet);
    bool simIsRunning();
    void updateGpsSim();
    void updateOrientationSim();
    void updateCameraSim();
    bool getSimulatorErrors(std::vector<std::string>& errorMsgs);

    void             stopSim();
    void             startSim();
    bool             isPausedSim();
    void             pauseSim();
    void             resumeSim();
    SENSTimePt       simTime();
    SENSMicroseconds passedSimTime();

    SENSGps*           gps() const { return _gpsRef; }
    SENSOrientation*   orientation() const { return _orientationRef; }
    SENSCamera*        camera() const { return _cameraRef; }
    const std::string& simDataDir() const { return _simDataDir; }

    bool recordGps         = false;
    bool recordOrientation = false;
    bool recordCamera      = false;

    bool simulateGps         = false;
    bool simulateOrientation = false;
    bool simulateCamera      = false;

private:
    void          restoreInputSensors();
    SENSRecorder* recorder();

    std::unique_ptr<SENSSimulator> _simulator;
    std::unique_ptr<SENSRecorder>  _recorder;

    SENSGps*         _gpsIn;
    SENSOrientation* _orientationIn;
    SENSCamera*      _cameraIn;

    SENSGps*&         _gpsRef;
    SENSOrientation*& _orientationRef;
    SENSCamera*&      _cameraRef;

    std::string _simDataDir;

    //callback to inform about new camera parameter (e.g. intrinsics for scene and tracking)
    std::function<void(void)> _cameraParametersChangedCB;
};

#endif
