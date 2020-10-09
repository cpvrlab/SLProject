#ifndef SENS_SIMHELPER_H
#define SENS_SIMHELPER_H

#include <sens/SENSSimulator.h>
#include <sens/SENSRecorder.h>

class SENSSimHelper
{
public:
    //The camera pointer has to be used as reference
    SENSSimHelper(SENSGps*&                 gps,
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

    ~SENSSimHelper()
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

    bool startRecording()
    {
        if (!_recorder)
            return false;

        //sensors should have been activated before
        return _recorder->start();
    }

    void stopRecording()
    {
        if (_recorder)
            _recorder->stop();
    }

    void toggleGpsRecording()
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

    void toggleOrientationRecording()
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

    void toggleCameraRecording()
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

    SENSRecorder* recorder()
    {
        if (!_recorder)
            _recorder = std::make_unique<SENSRecorder>(_simDataDir);
        return _recorder.get();
    }

    bool recorderIsRunning()
    {
        if(_recorder)
            return _recorder->isRunning();
        else
            return false;
    }
    
    void resetRecorder()
    {
        if (_recorder)
            _recorder.reset();

        recordGps         = false;
        recordOrientation = false;
        recordCamera      = false;
    }

    bool canSimGps()
    {
        if (_simulator && _simulator->getGpsSensorPtr())
            return true;
        else
            return false;
    }

    bool canSimOrientation()
    {
        if (_simulator && _simulator->getOrientationSensorPtr())
            return true;
        else
            return false;
    }

    bool canSimCamera()
    {
        if (_simulator && _simulator->getCameraSensorPtr())
            return true;
        else
            return false;
    }

    void initSimulator(const std::string& simDataSet)
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

    bool simIsRunning()
    {
        if (_simulator && _simulator->isRunning())
            return true;
        else
            return false;
    }

    void stopSim()
    {
        //stop running sensors
        if (_gpsRef && _gpsRef->isRunning())
            _gpsRef->stop();

        if (_orientationRef && _orientationRef->isRunning())
            _orientationRef->stop();

        if (_cameraRef && _cameraRef->started())
            _cameraRef->stop();
    }

    void startSim()
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
                auto                              sc           = currCamProps->streamConfigs().front();
                //first start the camera, the intrinsic is valid afterwards
                _cameraRef->start(currCamProps->deviceId(),
                                  sc,
                                  {sc.widthPix, sc.heightPix},
                                  false,
                                  false,
                                  true);
                if (_cameraParametersChangedCB)
                    _cameraParametersChangedCB();
            }
        }
    }

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
