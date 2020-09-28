#ifndef SENS_RECORDER_H
#define SENS_RECORDER_H

#include <string>
#include <thread>
#include <atomic>
#include <deque>
#include <chrono>
#include <utility>
#include <mutex>
#include <fstream>

#include <sens/SENSGps.h>
#include <sens/SENSOrientation.h>
#include <sens/SENSCamera.h>
#include <Utils.h>

typedef std::chrono::high_resolution_clock             HighResClock;
typedef std::chrono::high_resolution_clock::time_point HighResTimePoint;

template<typename T>
class SENSRecorderDataHandler
{
public:
    SENSRecorderDataHandler(const std::string& name, const std::string& outputDir, std::function<void(T&, ofstream&)> writeDataToFile)
      : _name(name),
        _outputDir(outputDir),
        _writeDataToFile(writeDataToFile)
    {
        assert(_writeDataToFile);
    }

    ~SENSRecorderDataHandler()
    {
        stop();
    }

    void stop()
    {
        _stop = true;
        _condVar.notify_one();

        if (_thread.joinable())
            _thread.join();
        _stop = false;
    }

    void start()
    {
        stop();
        //start writer thread
        _thread = std::thread(&SENSRecorderDataHandler::store, this);
    }

    void store()
    {
        //open file
        std::string fileName = _outputDir + _name + ".txt";
        ofstream    file;
        file.open(fileName);
        if (file.is_open())
        {
            while (true)
            {
                std::unique_lock<std::mutex> lock(_mutex);

                _condVar.wait(lock, [&] { return (_stop == true || _queue.size() != 0); });
                if (_stop)
                    break;

                std::deque<T> localQueue;
                while (_queue.size())
                {
                    localQueue.push_back(_queue.front());
                    _queue.pop_front();
                }

                lock.unlock();

                //write data
                while (localQueue.size())
                {
                    _writeDataToFile(localQueue.front(), file);
                    localQueue.pop_front();
                }
            }
        }
        //close file
        file.close();
    }

    void add(T&& item)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _queue.push_back(std::move(item));
        lock.unlock();
        _condVar.notify_one();
    }

private:
    std::deque<T>           _queue;
    std::mutex              _mutex;
    std::condition_variable _condVar;

    std::atomic_bool _stop{false};
    std::thread      _thread;

    std::string                        _name;
    std::string                        _outputDir;
    std::function<void(T&, ofstream&)> _writeDataToFile;
};

class SENSRecorder : public SENSGpsListener
  , public SENSOrientationListener //, public SENSCameraListener
{
    using GpsInfo         = std::pair<SENSGps::Location, HighResTimePoint>;
    using OrientationInfo = std::pair<SENSOrientation::Quat, HighResTimePoint>;
    using FrameInfo       = std::pair<SENSFramePtr, HighResTimePoint>;

public:
    SENSRecorder(const std::string& outputDir)
    {
        if (Utils::dirExists(outputDir))
        {

            std::string recordDir = Utils::unifySlashes(outputDir) + Utils::getDateTime2String() + "_SENSRecorder/";
            Utils::makeDir(recordDir);
            if (Utils::dirExists(recordDir))
            {
                auto gpsWriteDataFct = [](GpsInfo& data, ofstream& file) {
                    using namespace std::chrono;

                    file << time_point_cast<microseconds>(data.second).time_since_epoch().count() << " "
                         << data.first.latitudeDEG << " "
                         << data.first.longitudeDEG << " "
                         << data.first.altitudeM << " "
                         << data.first.accuracyM << "\n";
                };
                auto orientationWriteDataFct = [](OrientationInfo& data, ofstream& file) {
                    using namespace std::chrono;
                    long timePtUs = time_point_cast<microseconds>(data.second).time_since_epoch().count();

                    //reading (https://stackoverflow.com/questions/31255486/c-how-do-i-convert-a-stdchronotime-point-to-long-and-back)
                    //long readTimePt;
                    //microseconds readTimePtUs(readTimePt);
                    //time_point<high_resolution_clock> dt(readTimePt);

                    file << timePtUs << " "
                         << data.first.quatX << " "
                         << data.first.quatY << " "
                         << data.first.quatZ << " "
                         << data.first.quatW << "\n";
                };

                _gpsDataHandler         = new SENSRecorderDataHandler<GpsInfo>("gps", recordDir, gpsWriteDataFct);
                _orientationDataHandler = new SENSRecorderDataHandler<OrientationInfo>("orientation", recordDir, orientationWriteDataFct);
                //_cameraDataHandler = SENSRecorderDataHandler<FrameInfo>("camera");
            }
            else
                Utils::log("SENS", "SENSRecorder: Could not create output directory: %s", outputDir.c_str());
        }
        else
            Utils::log("SENS", "SENSRecorder: Directory does not exist: %s", outputDir.c_str());
    }
    ~SENSRecorder()
    {
        stop();

        if (_gpsDataHandler)
            delete _gpsDataHandler;
        if (_orientationDataHandler)
            delete _orientationDataHandler;
        //delete _cameraDataHandler;
    }

    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    void activateGps(SENSGps* sensor)
    {
        if (_running)
            return;

        if (sensor)
        {
            deactivateGps();
            _gps = sensor;
        }
    }
    void deactivateGps()
    {
        if (_running)
            return;

        if (_gps)
            _gps = nullptr;
    }
    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    void activateOrientation(SENSOrientation* sensor)
    {
        if (_running)
            return;

        if (sensor)
        {
            deactivateOrientation();
            _orientation = sensor;
        }
    }

    void deactivateOrientation()
    {
        if (_running)
            return;

        if (_orientation)
            _orientation = nullptr;
    }

    /*
    //try to activate sensors before starting them, otherwise listener registration may become a threading problem
    void activateCamera(SENSCamera* sensor)
    {
        if(_running)
            return;
        
        if(sensor)
        {
            deactivateCamera();
            _camera = sensor;
            _camera->registerListener(this);
        }
    }
    
    void deactivateCamera()
    {
        if(_running)
            return;
        
        if(_camera)
        {
            _camera->unregisterListener(this);
            _camera = nullptr;
        }
    }
     */

    void start()
    {
        if (_running)
            return;

        _running = true;

        if (_gps)
        {
            _gps->registerListener(this);
            if (_gpsDataHandler)
                _gpsDataHandler->start();
        }

        if (_orientation)
        {
            _orientation->registerListener(this);
            if (_orientationDataHandler)
                _orientationDataHandler->start();
        }
    }

    void stop()
    {
        if (_gps)
        {
            if (_gpsDataHandler)
                _gpsDataHandler->stop();
            _gps->unregisterListener(this);
        }

        if (_orientation)
        {
            if (_orientationDataHandler)
                _orientationDataHandler->stop();
            _orientation->unregisterListener(this);
        }

        _running = false;
    }

private:
    void onGps(const SENSGps::Location& loc) override
    {
        auto newData = std::make_pair(loc, HighResClock::now());
        if (_gpsDataHandler)
            _gpsDataHandler->add(std::move(newData));
    }

    void onOrientation(const SENSOrientation::Quat& ori) override
    {
        auto newData = std::make_pair(ori, HighResClock::now());
        if (_orientationDataHandler)
            _orientationDataHandler->add(std::move(newData));
    }

    /*
    void onFrame(SENSFramePtr frame) override
    {
        auto newData = std::make_pair(frame, getTime()));
        _cameraDataHandler.add(newData);
    }
     */

    std::string _outputDir;

    SENSRecorderDataHandler<GpsInfo>*         _gpsDataHandler         = nullptr;
    SENSRecorderDataHandler<OrientationInfo>* _orientationDataHandler = nullptr;
    SENSRecorderDataHandler<FrameInfo>*       _cameraDataHandler      = nullptr;

    SENSGps*         _gps         = nullptr;
    SENSOrientation* _orientation = nullptr;
    SENSCamera*      _camera      = nullptr;

    std::atomic_bool _running{false};
};

#endif
