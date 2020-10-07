#ifndef SENS_SIMULATED_H
#define SENS_SIMULATED_H

#include <functional>
#include <memory>
#include <chrono>

#include <Utils.h>
#include <sens/SENSSimClock.h>
#include <sens/SENSCamera.h>
#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>

class SENSSimulator;

//-----------------------------------------------------------------------------
/*! SENSSimulatedBase
This ia pure virtual base class to control SENSSimulated implementations via a common interface in SENSSimulator
 */
class SENSSimulatedBase
{
public:
    virtual ~SENSSimulatedBase() {}
    //!indicates if simulation thread is running
    virtual bool isThreadRunning() const = 0;

    //!get the first time point in simulation sensor data
    //virtual const SENSTimePt& firstTimePt() = 0;
    //!called by SENSSimulator to set the common simulation start time
    //virtual void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt) = 0;
};

//-----------------------------------------------------------------------------
/*! SENSSimulated
Implements the sensor simulation backend. SENSSimulated runs a thread (see feedSensor()) that calls feedSensorData() interface
function which feeds the next data value to a sensor (e.g. SENSGps). The current simulation time is retrieved from
the SENSSimulator clock and depending on this the next data value is selected and fed a the respective time.
This class contains common functionality for SENSSimulated implementations.
 */
template<typename T>
class SENSSimulated : public SENSSimulatedBase
{
public:
    virtual ~SENSSimulated() {}

protected:
    using StartSimCB         = std::function<void(void)>;
    using SensorSimStoppedCB = std::function<void(void)>;

    //!Transfer callbacks to SENSSimulator during construction. The SENSSimulator is informed by
    //!these if the SENSSimulated state changes (start, stop). We use callbacks to avoid circular dependencies.
    SENSSimulated(StartSimCB                              startSimCB,
                  SensorSimStoppedCB                      sensorSimStoppedCB,
                  std::vector<std::pair<SENSTimePt, T>>&& data,
                  const SENSSimClock&                     clock);

    //!start the sensor simulation thread
    void startSim();
    //!stop the sensor simulation thread
    void stopSim();

    //!called by SENSSimulator to set the common simulation start time
    //void setCommonSimStartTimePt(const SENSTimePt& commonSimStartTimePt) override;
    //!get the first time point in simulation sensor data
    //const SENSTimePt& firstTimePt() override;
    bool              isThreadRunning() const override { return _threadIsRunning; }

    //!feed new sensor data to sensor
    virtual void feedSensorData(const int counter) = 0;
    //!prepare things that may take some time for the next writing of sensor data
    //!(e.g. for video reading we can already read the frame and do decoding and maybe it also blocks for some reasons..)
    virtual void prepareSensorData(const int counter){};

private:
    //!thread run routine to feed sensor with data.
    void feedSensor(/*const SENSTimePt startTimePt, const SENSTimePt simStartTimePt*/);

protected:
    std::vector<std::pair<SENSTimePt, T>> _data;

    //inform simulator that sensor was stopped
    SensorSimStoppedCB _sensorSimStoppedCB;
    StartSimCB         _startSimCB;

    std::thread             _thread;
    std::condition_variable _condVar;
    std::mutex              _mutex;
    bool                    _stop = false;

    SENSTimePt _commonSimStartTimePt;

    bool _threadIsRunning = false;

    const SENSSimClock& _clock;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedGps
SENSSimulated implementation for GPS sensor simulation. Implements the SENSGps interface
 to make it a full gps sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedGps : public SENSGps
  , public SENSSimulated<SENSGps::Location>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedGps();

    bool start() override;
    void stop() override;

private:
    //only SENSSimulator can instantiate
    SENSSimulatedGps(StartSimCB                                              startSimCB,
                     SensorSimStoppedCB                                      sensorSimStoppedCB,
                     std::vector<std::pair<SENSTimePt, SENSGps::Location>>&& data,
                     const SENSSimClock&                                     clock);

    void feedSensorData(const int counter) override;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedOrientation
SENSSimulated implementation for Orientation sensor simulation. Implements the SENSOrientation interface
 to make it a full orientation sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedOrientation : public SENSOrientation
  , public SENSSimulated<SENSOrientation::Quat>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedOrientation();

    bool start() override;
    void stop() override;

private:
    SENSSimulatedOrientation(StartSimCB                                                  startSimCB,
                             SensorSimStoppedCB                                          sensorSimStoppedCB,
                             std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>&& data,
                             const SENSSimClock&                                         clock);

    void feedSensorData(const int counter) override;
};

//-----------------------------------------------------------------------------
/*! SENSSimulatedCamera
SENSSimulated implementation for Camera sensor simulation. Implements the SENSCameraBase interface
to make it a full camera sensor and the SENSSimulated interface for the sensor simulation backend.
 */
class SENSSimulatedCamera : public SENSCameraBase
  , public SENSSimulated<int>
{
    friend class SENSSimulator;

public:
    ~SENSSimulatedCamera();

    const SENSCameraConfig& start(std::string                   deviceId,
                                  const SENSCameraStreamConfig& streamConfig,
                                  cv::Size                      imgBGRSize           = cv::Size(),
                                  bool                          mirrorV              = false,
                                  bool                          mirrorH              = false,
                                  bool                          convToGrayToImgManip = false,
                                  int                           imgManipWidth        = -1,
                                  bool                          provideIntrinsics    = true,
                                  float                         fovDegFallbackGuess  = 65.f) override;

    void stop() override;

    SENSFramePtr                 latestFrame() override;
    const SENSCaptureProperties& captureProperties() override;

private:
    SENSSimulatedCamera(StartSimCB                                startSimCB,
                        SensorSimStoppedCB                        sensorSimStoppedCB,
                        std::vector<std::pair<SENSTimePt, int>>&& data,
                        std::string                               videoFileName,
                        const SENSSimClock&                       clock);

    void feedSensorData(const int counter) override;
    void prepareSensorData(const int counter) override;

    std::string      _videoFileName;
    cv::VideoCapture _cap;

    cv::Mat    _preparedFrame;
    int        _preparedFrameIndex = -1;
    cv::Mat    _frame;
    std::mutex _frameMutex;
};

#endif
