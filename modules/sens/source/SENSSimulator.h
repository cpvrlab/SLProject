#ifndef SENS_SIMULATOR_H
#define SENS_SIMULATOR_H

#include <functional>
#include <memory>
#include <chrono>

#include <Utils.h>
#include <SENSSimulated.h>
#include <SENSSimClock.h>

/*!SENSSimulator
 This class is owner of SENSSimulated sensor instances. It loads special sensor data from
 input simulation directory transferred during construction. Depending on if the directory content is valid,
 a SENSSimulated implementation is instantiated and initialized with loaded sensor data.
 You can get a sensor pointer to a simulated sensor by calling the getter get..SensorPtr() method and
 use it like the real sensor, as it implements the original sensor interface.
 The SENSSimulator is maintainer of the current simulation time. Simulated sensor values are selected
 depending on this time. The idea is to provide sensor data exactly as it was recorded.
 ATTENTION: Simulation time starts, as soon as one simulated sensor was started and is resetted when all simulated
 sensors are stopped.
 */
class SENSSimulator
{
public:
    //!ctor, transfer directory containing sensor data that was recorded with SENSRecorder
    SENSSimulator(const std::string& simDirName);
    ~SENSSimulator();

    //!get sensor pointer of simulated gps sensor and use it like a normal sensor (valid if not null)
    SENSSimulatedGps* getGpsSensorPtr() { return getActiveSensor<SENSSimulatedGps>(); }
    //!get sensor pointer of simulated orientation sensor and use it like a normal sensor (valid if not null)
    SENSSimulatedOrientation* getOrientationSensorPtr() { return getActiveSensor<SENSSimulatedOrientation>(); }
    //!get sensor pointer of simulated camera sensor and use it like a normal sensor (valid if not null)
    SENSSimulatedCamera* getCameraSensorPtr() { return getActiveSensor<SENSSimulatedCamera>(); }
    //!indicates if simulator is currently running
    bool isRunning() const { return _running; }

    bool getSimulatorErrors(std::vector<std::string>& errorMsgs);

    //!pause simulation time
    void pause();
    bool isPaused();
    //!resume simulation time
    void resume();
    //!reset simulation time
    SENSTimePt now();
    //!get passed simulation time
    SENSMicroseconds passedTime();

private:
    template<typename T>
    T* getActiveSensor()
    {
        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            SENSSimulatedBase* base    = _activeSensors[i].get();
            T*                 derived = dynamic_cast<T*>(base);
            if (derived)
                return derived;
        }

        return nullptr;
    }
    //!callback called from SENSSimulated when the simulated sensor was started
    void onStart();
    //!callback called from SENSSimulated when the simulated sensor was stopped
    void onSensorSimStopped();

    //!load data from file and instantiate sensor if valid
    void loadGpsData(const std::string& dirName, std::vector<std::pair<SENSTimePt, SENSGps::Location>>& data);
    //!load data from file and instantiate sensor if valid
    void loadOrientationData(const std::string& dirName, std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>& data);
    //!load data from file and instantiate sensor if valid
    void loadCameraData(const std::string& dirName, std::vector<std::pair<SENSTimePt, int>>& data, std::string& videoFileName, SENSCameraConfig& cameraConfig);

    //!list of currently activated sensors. Which sensors are activated depends on the content of simulation directory
    std::vector<std::unique_ptr<SENSSimulatedBase>> _activeSensors;

    std::unique_ptr<SENSSimClock> _clock;
    //!flags if simulator is currently running
    std::atomic_bool _running{false};
};

#endif
