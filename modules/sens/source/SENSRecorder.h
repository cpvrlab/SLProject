#ifndef SENS_RECORDER_H
#define SENS_RECORDER_H

#include <string>
#include <atomic>

#include <SENSRecorderDataHandler.h>

//-----------------------------------------------------------------------------
/*! SENSRecorder
 This class records implemented sensor classes to file. The output directory has to be transfered during instantiation.
 Activate or deactivate a sensor by calling activate or deactivate methods. The functions return true on success, a
 change is only possible if the recorder is not running (use isRunning method).
 Call start() and stop() to start or stop recording of sensors.
 The SENSRecorder listens to sensors and informs SENSRecorderDataHandler backends about new data.
 The SENSRecorderDataHandler stores values to file.
 */
class SENSRecorder : public SENSGpsListener
  , public SENSOrientationListener
  , public SENSCameraListener
{
public:
    SENSRecorder(const std::string& outputDir);
    ~SENSRecorder();

    bool activateGps(SENSGps* sensor);
    bool deactivateGps();
    bool activateOrientation(SENSOrientation* sensor);
    bool deactivateOrientation();
    bool activateCamera(SENSCamera* sensor);
    bool deactivateCamera();

    //!Start the recording of activated sensors (Registers listeners and starts SENSRecorderDataHandler backends).
    //!Returns true on success. Possible reasons for fail are: Output directory does not exist or recorder is already running.
    bool start();
    //!Stops a running recording session.
    void stop();

    bool getGpsHandlerError(std::string& errorMsg);
    bool getOrientationHandlerError(std::string& errorMsg);
    bool getCameraHandlerError(std::string& errorMsg);

    bool               isRunning() const { return _running; }
    const std::string& outputDir() const { return _outputDir; }

private:
    //!gps listener callback
    void onGps(const SENSTimePt& timePt, const SENSGps::Location& loc) override;
    //!orientation listener callback
    void onOrientation(const SENSTimePt& timePt, const SENSOrientation::Quat& ori) override;
    //!camera listener callback
    void onFrame(const SENSTimePt& timePt, cv::Mat frame) override;
    void onCameraConfigChanged(const SENSCameraConfig& config) override;

    std::string _outputDir;

    SENSGpsRecorderDataHandler*         _gpsDataHandler         = nullptr;
    SENSOrientationRecorderDataHandler* _orientationDataHandler = nullptr;
    SENSCameraRecorderDataHandler*      _cameraDataHandler      = nullptr;

    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;
    SENSCamera*      _camera      = nullptr;

    std::atomic_bool _running{false};
};

#endif
