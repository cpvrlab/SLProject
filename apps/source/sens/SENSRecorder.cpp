#include "SENSRecorder.h"
#include <Utils.h>

SENSRecorder::SENSRecorder(const std::string& outputDir)
  : _outputDir(outputDir)
{
    if (Utils::dirExists(outputDir))
    {
        _gpsDataHandler         = new SENSGpsRecorderDataHandler();
        _orientationDataHandler = new SENSOrientationRecorderDataHandler();
        _cameraDataHandler      = new SENSCameraRecorderDataHandler();
    }
    else
        Utils::log("SENS", "SENSRecorder: Directory does not exist: %s", outputDir.c_str());
}

SENSRecorder::~SENSRecorder()
{
    stop();

    if (_gpsDataHandler)
        delete _gpsDataHandler;
    if (_orientationDataHandler)
        delete _orientationDataHandler;
    if (_cameraDataHandler)
        delete _cameraDataHandler;
}

//try to activate sensors before starting them, otherwise listener registration may become a threading problem
bool SENSRecorder::activateGps(SENSGps* sensor)
{
    if (!_running && sensor)
    {
        deactivateGps();
        _gps = sensor;
        return true;
    }
    else
        return false;
}
bool SENSRecorder::deactivateGps()
{
    if (!_running && _gps)
    {
        _gps = nullptr;
        return true;
    }
    else
        return false;
}
//try to activate sensors before starting them, otherwise listener registration may become a threading problem
bool SENSRecorder::activateOrientation(SENSOrientation* sensor)
{
    if (!_running && sensor)
    {
        deactivateOrientation();
        _orientation = sensor;
        return true;
    }
    else
        return false;
}

bool SENSRecorder::deactivateOrientation()
{
    if (!_running && _orientation)
    {
        _orientation = nullptr;
        return true;
    }
    else
        return true;
}

//try to activate sensors before starting them, otherwise listener registration may become a threading problem
bool SENSRecorder::activateCamera(SENSCamera* sensor)
{
    if (!_running && sensor)
    {
        deactivateCamera();
        _camera = sensor;
        return true;
    }
    else
        return false;
}

bool SENSRecorder::deactivateCamera()
{
    if (!_running && _camera)
    {
        _camera = nullptr;
        return true;
    }
    else
        return true;
}

bool SENSRecorder::start()
{
    if (_running)
        return false;

    std::string recordDir = Utils::unifySlashes(_outputDir) + Utils::getDateTime2String() + "_SENSRecorder/";
    if (!Utils::makeDir(recordDir))
    {
        Utils::log("SENS", "SENSRecorder start: could not create record directory: %s", recordDir.c_str());
        return false;
    }

    if (_gps && _gpsDataHandler)
    {
        _gpsDataHandler->start(recordDir);
        _gps->registerListener(this);
        _running = true;
    }

    if (_orientation && _orientationDataHandler)
    {
        _orientationDataHandler->start(recordDir);
        _orientation->registerListener(this);
        _running = true;
    }

    if (_camera && _cameraDataHandler)
    {
        _cameraDataHandler->start(recordDir);
        _camera->registerListener(this);
        _running = true;
    }

    return _running;
}

void SENSRecorder::stop()
{
    if (_gps)
    {
        _gps->unregisterListener(this);
        if (_gpsDataHandler)
            _gpsDataHandler->stop();
    }

    if (_orientation)
    {
        _orientation->unregisterListener(this);
        if (_orientationDataHandler)
            _orientationDataHandler->stop();
    }

    if (_camera)
    {
        _camera->unregisterListener(this);
        if (_cameraDataHandler)
            _cameraDataHandler->stop();
    }

    _running = false;
}

bool SENSRecorder::getGpsHandlerError(std::string& errorMsg)
{
    if (_gpsDataHandler && _gpsDataHandler->getErrorMsg(errorMsg))
        return true;
    else
        return false;
}

bool SENSRecorder::getOrientationHandlerError(std::string& errorMsg)
{
    if (_orientationDataHandler && _orientationDataHandler->getErrorMsg(errorMsg))
        return true;
    else
        return false;
}

bool SENSRecorder::getCameraHandlerError(std::string& errorMsg)
{
    if (_cameraDataHandler && _cameraDataHandler->getErrorMsg(errorMsg))
        return true;
    else
        return false;
}

void SENSRecorder::onGps(const SENSTimePt& timePt, const SENSGps::Location& loc)
{
    auto newData = std::make_pair(loc, timePt);
    if (_gpsDataHandler)
        _gpsDataHandler->add(std::move(newData));
}

void SENSRecorder::onOrientation(const SENSTimePt& timePt, const SENSOrientation::Quat& ori)
{
    auto newData = std::make_pair(ori, timePt);
    if (_orientationDataHandler)
        _orientationDataHandler->add(std::move(newData));
}

void SENSRecorder::onFrame(const SENSTimePt& timePt, cv::Mat frame)
{
    auto newData = std::make_pair(frame, timePt);
    if (_cameraDataHandler)
        _cameraDataHandler->add(std::move(newData));
}

void SENSRecorder::onCameraConfigChanged(const SENSCameraConfig& config)
{
    if (_cameraDataHandler)
        _cameraDataHandler->updateConfig(config);
}
