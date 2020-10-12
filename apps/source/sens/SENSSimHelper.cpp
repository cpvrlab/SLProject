#include "SENSSimHelper.h"

SENSSimHelper::SENSSimHelper(SENSGps*&                 gps,
                             SENSOrientation*&         orientation,
                             SENSCamera*&              camera,
                             const std::string&        simDataDir,
                             std::function<void(void)> cameraParametersChangedCB)
  : _gpsIn(gps),
    _orientationIn(orientation),
    _cameraIn(camera),
    _gpsRef(gps),
    _orientationRef(orientation),
    _cameraRef(camera),
    _simDataDir(Utils::unifySlashes(simDataDir)),
    _cameraParametersChangedCB(cameraParametersChangedCB)
{
}

SENSSimHelper::~SENSSimHelper()
{
    resetRecorder();

    //restore input sensors and state
    if (_gpsRef != _gpsIn)
    {
        if (_gpsRef && _gpsRef->isRunning())
            _gpsRef->stop();
        _gpsRef = _gpsIn;
    }

    if (_orientationRef != _orientationIn)
    {
        if (_orientationRef && _orientationRef->isRunning())
            _orientationRef->stop();
        _orientationRef = _orientationIn;
    }

    if (_cameraRef != _cameraIn)
    {
        if (_cameraRef && _cameraRef->started())
            _cameraRef->stop();
        _cameraRef = _cameraIn;
    }
}

bool SENSSimHelper::startRecording()
{
    if (!_recorder)
        return false;

    //sensors should have been activated before
    return _recorder->start();
}

void SENSSimHelper::stopRecording()
{
    if (_recorder)
        _recorder->stop();
}

void SENSSimHelper::toggleGpsRecording()
{
    if (recordGps)
    {
        if (!recorder()->activateGps(_gpsRef))
            recordGps = !recordGps;
    }
    else
    {
        if (!recorder()->deactivateGps())
            recordGps = !recordGps;
    }
}

void SENSSimHelper::toggleOrientationRecording()
{
    if (recordOrientation)
    {
        if (!recorder()->activateOrientation(_orientationRef))
            recordOrientation = !recordOrientation;
    }
    else
    {
        if (!recorder()->deactivateOrientation())
            recordOrientation = !recordOrientation;
    }
}

void SENSSimHelper::toggleCameraRecording()
{
    if (recordCamera)
    {
        if (!recorder()->activateCamera(_cameraRef))
            recordCamera = !recordCamera;
    }
    else
    {
        if (!recorder()->deactivateOrientation())
            recordCamera = !recordCamera;
    }
}

SENSRecorder* SENSSimHelper::recorder()
{
    if (!_recorder)
        _recorder = std::make_unique<SENSRecorder>(_simDataDir);
    return _recorder.get();
}

bool SENSSimHelper::recorderIsRunning()
{
    if (_recorder)
        return _recorder->isRunning();
    else
        return false;
}

void SENSSimHelper::resetRecorder()
{
    if (_recorder)
        _recorder.reset();

    recordGps         = false;
    recordOrientation = false;
    recordCamera      = false;
}

bool SENSSimHelper::canSimGps()
{
    if (_simulator && _simulator->getGpsSensorPtr())
        return true;
    else
        return false;
}

bool SENSSimHelper::canSimOrientation()
{
    if (_simulator && _simulator->getOrientationSensorPtr())
        return true;
    else
        return false;
}

bool SENSSimHelper::canSimCamera()
{
    if (_simulator && _simulator->getCameraSensorPtr())
        return true;
    else
        return false;
}

void SENSSimHelper::initSimulator(const std::string& simDataSet)
{
    resetRecorder();
    //stop running simulations
    if (canSimGps() && simulateGps)
        _simulator->getGpsSensorPtr()->stop();
    if (canSimOrientation() && simulateOrientation)
        _simulator->getOrientationSensorPtr()->stop();
    if (canSimCamera() && simulateCamera)
        _simulator->getCameraSensorPtr()->stop();

    if (_simulator)
        _simulator.reset();

    _simulator = std::make_unique<SENSSimulator>(_simDataDir + simDataSet);
}

bool SENSSimHelper::simIsRunning()
{
    if (_simulator && _simulator->isRunning())
        return true;
    else
        return false;
}

void SENSSimHelper::stopSim()
{
    //stop running sensors
    if (_gpsRef && _gpsRef->isRunning())
        _gpsRef->stop();

    if (_orientationRef && _orientationRef->isRunning())
        _orientationRef->stop();

    if (_cameraRef && _cameraRef->started())
        _cameraRef->stop();
}

void SENSSimHelper::startSim()
{
    //stop running sensors
    stopSim();

    //start possible simulated sensors
    if (canSimGps() && simulateGps)
    {
        _gpsRef = _simulator->getGpsSensorPtr();
        _gpsRef->start();
    }

    if (canSimOrientation() && simulateOrientation)
    {
        _orientationRef = _simulator->getOrientationSensorPtr();
        _orientationRef->start();
    }

    if (canSimCamera() && simulateCamera)
    {
        _cameraRef = _simulator->getCameraSensorPtr();
        //there is only one prop:
        const SENSCaptureProperties& capProps = _cameraRef->captureProperties();
        if (capProps.size())
        {
            const SENSCameraDeviceProperties* currCamProps = &capProps.front();
            auto                              streamConfig = currCamProps->streamConfigs().front();
            //first start the camera, the intrinsic is valid afterwards
            _cameraRef->start(currCamProps->deviceId(),
                              streamConfig,
                              true,
                              65.f);
            if (_cameraParametersChangedCB)
                _cameraParametersChangedCB();
        }
    }
}

bool SENSSimHelper::isPausedSim()
{
    if(_simulator && _simulator->isPaused())
        return true;
    else
        return false;
}

void SENSSimHelper::pauseSim()
{
    if(_simulator)
        _simulator->pause();
}

void SENSSimHelper::resumeSim()
{
    if(_simulator)
        _simulator->resume();
}

SENSTimePt SENSSimHelper::simTime()
{
    if (_simulator)
        return _simulator->now();
    else
        return SENSClock::now();
}
SENSMicroseconds SENSSimHelper::passedSimTime()
{
    if (_simulator)
        return _simulator->passedTime();
    else
        return SENSMicroseconds(0);
}

