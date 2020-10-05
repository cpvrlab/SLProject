#include "SENSOrientation.h"
#include <Utils.h>

SENSOrientation::Quat SENSOrientation::getOrientation()
{
    const std::lock_guard<std::mutex> lock(_orientationMutex);
    return _orientation;
}

void SENSOrientation::setOrientation(SENSOrientation::Quat orientation)
{
    //estimate time before running into lock
    SENSTimePt timePt = SENSClock::now();
    
    {
        const std::lock_guard<std::mutex> lock(_orientationMutex);
        _orientation = orientation;
        _timePt = timePt;
    }
    
    {
        std::lock_guard<std::mutex> lock(_listenerMutex);
        for(SENSOrientationListener* l : _listeners)
            l->onOrientation(timePt, orientation);
    }
}

void SENSOrientation::registerListener(SENSOrientationListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    if(std::find(_listeners.begin(), _listeners.end(), listener) == _listeners.end())
        _listeners.push_back(listener);
}
void SENSOrientation::unregisterListener(SENSOrientationListener* listener)
{
    std::lock_guard<std::mutex> lock(_listenerMutex);
    for(auto it = _listeners.begin(); it != _listeners.end(); ++it)
    {
        if(*it == listener)
        {
            _listeners.erase(it);
            break;
        }
    }
}

/**********************************************************************/

SENSDummyOrientation::~SENSDummyOrientation()
{
    stopSimulation();
}

bool SENSDummyOrientation::start()
{
    if(_running)
        return false;
    
    Utils::log("SENSDummyOrientation", "start");
    startSimulation();
    _running = true;
    return true;
}

void SENSDummyOrientation::stop()
{
    Utils::log("SENSDummyOrientation", "stop");
    stopSimulation();
    _running = false;
}

void SENSDummyOrientation::startSimulation()
{
    _stop   = false;
    _thread = std::thread(&SENSDummyOrientation::run, this);
}

void SENSDummyOrientation::stopSimulation()
{
    _stop = true;
    if (_thread.joinable())
        _thread.join();
}

void SENSDummyOrientation::run()
{
    int i = 0;
    while (true)
    {
        std::this_thread::sleep_for(_intervalMS);
        if (_stop)
            break;

        i++;
        i = i % _dummyOrientations.size();

        Utils::log("SENSDummyOrientation", "run");
        setOrientation(_dummyOrientations[i]);
    }
}

void SENSDummyOrientation::setupDummyOrientations()
{
    _dummyOrientations = {
      {1.23345, 2.334534, 3.4, 4.5},
      {5.6, 6.7, 7.8, 8.9345345345}};
}

void SENSDummyOrientation::readFromFile(std::string fileName)
{
    string   line;
    ifstream file;
    file.open(fileName);
    if (file.is_open())
    {
        if (std::getline(file, line))
        {
            int interval = std::stoi(line);
            _intervalMS  = std::chrono::milliseconds(interval);
        }
        while (std::getline(file, line))
        {
            SENSOrientation::Quat    quat;
            std::vector<std::string> values;
            Utils::splitString(line, ',', values);
            if (values.size() == 4)
            {
                quat.quatX = std::stof(values[0]);
                quat.quatY = std::stof(values[1]);
                quat.quatZ = std::stof(values[2]);
                quat.quatW = std::stof(values[3]);
            }
            _dummyOrientations.push_back(quat);
        }
    }

    file.close();
}

/**********************************************************************/

SENSOrientationRecorder::SENSOrientationRecorder(SENSOrientation* sensor, std::string outputDir)
  : _sensor(sensor),
    _outputDir(Utils::unifySlashes(outputDir))
{
}

SENSOrientationRecorder::~SENSOrientationRecorder()
{
    stop();
}

bool SENSOrientationRecorder::start(std::chrono::milliseconds intervalMS)
{
    if (_running)
        stop();

    _intervalMS = intervalMS;

    if (_sensor)
    {
        _running = _sensor->start();
        if (_running)
            _thread = std::thread(&SENSOrientationRecorder::run, this);
    }

    return _running;
}

void SENSOrientationRecorder::stop()
{
    _stop = true;
    if (_thread.joinable())
        _thread.join();
    _running = false;
    _stop    = false;
}

void SENSOrientationRecorder::run()
{
    std::string fileName = _outputDir + Utils::getDateTime2String() + "_SENSOrientationRecorder.txt";
    ofstream    file;
    file.open(fileName);
    if (file.is_open())
    {
        //write interval in ms
        file << _intervalMS.count() << "\n";
        while (true)
        {
            if (_stop)
                break;

            std::this_thread::sleep_for(_intervalMS);

            //get value and write it to file
            SENSOrientation::Quat quat = _sensor->getOrientation();
            file << quat.quatX << "," << quat.quatY << "," << quat.quatZ << "," << quat.quatW << "\n";
        }
    }

    file.close();
    _running = false;
    _stop    = false;
}
