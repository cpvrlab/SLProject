#include "SENSSimulator.h"

SENSSimulator::~SENSSimulator()
{
}

bool SENSSimulator::getSimulatorErrors(std::vector<std::string>& errorMsgs)
{
    for (int i = 0; i < _activeSensors.size(); ++i)
    {
        SENSSimulatedBase* sensor = _activeSensors[i].get();
        std::string        msg;
        if (sensor->getErrorMsg(msg))
            errorMsgs.emplace_back(msg);
    }

    return errorMsgs.size() != 0;
}

void SENSSimulator::pause()
{
    if (_clock)
        _clock->pause();
}

bool SENSSimulator::isPaused()
{
    if (_clock)
        return _clock->isPaused();
    else
        return false;
}

void SENSSimulator::resume()
{
    if (_clock)
        _clock->resume();
}

SENSMicroseconds SENSSimulator::passedTime()
{
    SENSMicroseconds passedTime;
    if (_clock)
        passedTime = _clock->passedTime();
    return passedTime;
}

SENSTimePt SENSSimulator::now()
{
    SENSTimePt simNow;
    if (_clock)
        simNow = _clock->now();
    return simNow;
}

void findSimStartTimePt(SENSTimePt& simStartTimePt, bool& initialized, SENSTimePt tp)
{
    if (!initialized)
    {
        initialized    = true;
        simStartTimePt = tp;
    }
    else if (tp < simStartTimePt)
        simStartTimePt = tp;
}

SENSSimulator::SENSSimulator(const std::string& simDirName)
{
    try
    {
        //check directory content and enable simulated sensors depending on this
        std::string dirName = Utils::unifySlashes(simDirName);

        if (Utils::dirExists(dirName))
        {
            std::vector<std::pair<SENSTimePt, SENSGps::Location>> gpsData;
            loadGpsData(dirName, gpsData);
            std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>> orientationData;
            loadOrientationData(dirName, orientationData);
            std::vector<std::pair<SENSTimePt, int>> cameraData;
            std::string                             videoFileName;
            SENSCameraConfig                        cameraConfig;
            loadCameraData(dirName, cameraData, videoFileName, cameraConfig);

            //search for common sim time start point
            //start time point of simulation in the past
            SENSTimePt simStartTimePt;

            bool initialized = false;

            if (gpsData.size())
                findSimStartTimePt(simStartTimePt, initialized, gpsData.front().first);
            if (orientationData.size())
                findSimStartTimePt(simStartTimePt, initialized, orientationData.front().first);
            if (cameraData.size())
                findSimStartTimePt(simStartTimePt, initialized, cameraData.front().first);

            if (initialized)
            {
                //instantiate simulator clock
                _clock = std::make_unique<SENSSimClock>(SENSClock::now(), simStartTimePt);

                if (gpsData.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedGps>(
                      new SENSSimulatedGps(std::bind(&SENSSimulator::onStart, this),
                                           std::bind(&SENSSimulator::onSensorSimStopped, this),
                                           std::move(gpsData),
                                           *_clock)));
                }

                if (orientationData.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedOrientation>(
                      new SENSSimulatedOrientation(std::bind(&SENSSimulator::onStart, this),
                                                   std::bind(&SENSSimulator::onSensorSimStopped, this),
                                                   std::move(orientationData),
                                                   *_clock)));
                }

                if (cameraData.size())
                {
                    //make_unique has problems with fiendship and private constructor so we use unique_ptr
                    _activeSensors.push_back(std::unique_ptr<SENSSimulatedCamera>(
                      new SENSSimulatedCamera(std::bind(&SENSSimulator::onStart, this),
                                              std::bind(&SENSSimulator::onSensorSimStopped, this),
                                              std::move(cameraData),
                                              videoFileName,
                                              cameraConfig,
                                              *_clock)));
                }
            }
        }
        else
            Utils::log("SENS", "SENSSimulator: Directory does not exist: %s", simDirName.c_str());
    }
    catch (...)
    {
        Utils::log("SENS", "SENSSimulator: Exception while parsing sensor files");
    }
}

void SENSSimulator::onStart()
{
    if (!_running)
    {
        if (_clock)
            _clock->reset();
        _running = true;
    }
}

void SENSSimulator::onSensorSimStopped()
{
    //if no sensor is running anymore, we stop the simulation
    bool aSensorIsRunning = false;
    for (int i = 0; i < _activeSensors.size(); ++i)
    {
        if (_activeSensors[i]->isThreadRunning())
        {
            aSensorIsRunning = true;
            break;
        }
    }

    if (!aSensorIsRunning)
        _running = false;

    //THIS DOES NOT WORK BECAUSE WE WOULD JOIN A THREAD FROM THE CURRENT THREAD
    //assuming sensor implementations (e.g. SENSGps) are still in running we can restart the backend threads
    /*
    if (!_running && _repeat)
    {
        _clock.reset();
        for (int i = 0; i < _activeSensors.size(); ++i)
        {
            _activeSensors[i]->startSim();
        }
    }
    */
}

void SENSSimulator::loadGpsData(const std::string& dirName, std::vector<std::pair<SENSTimePt, SENSGps::Location>>& data)
{
    std::string gpsFileName = dirName + "gps.txt";
    //check if directory contains gps.txt
    if (Utils::fileExists(gpsFileName))
    {
        std::string line;
        ifstream    file(gpsFileName);
        if (file.is_open())
        {
            while (std::getline(file, line))
            {
                //cout << line << '\n';
                long long                readTimePt;
                SENSGps::Location        loc;
                std::vector<std::string> values;
                Utils::splitString(line, ' ', values);
                if (values.size() == 5)
                {
                    readTimePt       = std::stoll(values[0]);
                    loc.latitudeDEG  = std::stof(values[1]);
                    loc.longitudeDEG = std::stof(values[2]);
                    loc.altitudeM    = std::stof(values[3]);
                    loc.accuracyM    = std::stof(values[4]);

                    SENSMicroseconds readTimePtUs(readTimePt);
                    SENSTimePt       tPt(readTimePtUs);
                    data.push_back(std::make_pair(tPt, loc));
                }
            }
            file.close();
        }
        else
            Utils::log("SENS", "SENSSimulator: Unable to open file: %s", gpsFileName.c_str());
    }
}

void SENSSimulator::loadOrientationData(const std::string& dirName, std::vector<std::pair<SENSTimePt, SENSOrientation::Quat>>& data)
{
    std::string orientationFileName = dirName + "orientation.txt";
    //check if directory contains orientation.txt
    if (Utils::fileExists(orientationFileName))
    {
        std::string line;
        ifstream    file(orientationFileName);
        if (file.is_open())
        {
            while (std::getline(file, line))
            {
                //cout << line << '\n';
                long long                readTimePt;
                SENSOrientation::Quat    quat;
                std::vector<std::string> values;
                Utils::splitString(line, ' ', values);
                if (values.size() == 5)
                {
                    readTimePt = std::stoll(values[0]);
                    quat.quatX = std::stof(values[1]);
                    quat.quatY = std::stof(values[2]);
                    quat.quatZ = std::stof(values[3]);
                    quat.quatW = std::stof(values[4]);

                    SENSMicroseconds readTimePtUs(readTimePt);
                    SENSTimePt       tPt(readTimePtUs);
                    data.push_back(std::make_pair(tPt, quat));
                }
            }
            file.close();
        }
        else
            Utils::log("SENS", "SENSSimulator: Unable to open file: %s", orientationFileName.c_str());
    }
}

void SENSSimulator::loadCameraData(const std::string&                       dirName,
                                   std::vector<std::pair<SENSTimePt, int>>& data,
                                   std::string&                             videoFileName,
                                   SENSCameraConfig&                        cameraConfig)
{
    std::string textFileName   = dirName + "camera.txt";
    videoFileName              = dirName + "video.avi";
    std::string configFileName = dirName + "cameraConfig.json";

    //check if directory contains gps.txt
    if (Utils::fileExists(textFileName) && Utils::fileExists(videoFileName) && Utils::fileExists(configFileName))
    {
        //load config file
        {
            cv::FileStorage fs;
            fs.open(configFileName, cv::FileStorage::READ);
            if (fs.isOpened())
            {
                fs["deviceId"] >> cameraConfig.deviceId;
                fs["widthPix"] >> cameraConfig.streamConfig.widthPix;
                fs["heightPix"] >> cameraConfig.streamConfig.heightPix;
                fs["focalLengthPix"] >> cameraConfig.streamConfig.focalLengthPix;
                int focusMode;
                fs["focusMode"] >> focusMode;
                cameraConfig.focusMode = (SENSCameraFocusMode)focusMode;
                int facing;
                fs["facing"] >> facing;
                cameraConfig.facing = (SENSCameraFacing)facing;
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s ", configFileName.c_str());
        }

        //load frame info
        {
            std::string line;
            ifstream    file(textFileName);
            if (file.is_open())
            {
                while (std::getline(file, line))
                {
                    //cout << line << '\n';
                    long long                readTimePt;
                    int                      frameIndex;
                    std::vector<std::string> values;
                    Utils::splitString(line, ' ', values);
                    if (values.size() == 2)
                    {
                        readTimePt = std::stoll(values[0]);
                        frameIndex = std::stoi(values[1]);

                        SENSMicroseconds readTimePtUs(readTimePt);
                        SENSTimePt       tPt(readTimePtUs);
                        data.push_back(std::make_pair(tPt, frameIndex));
                    }
                }
                file.close();
            }
            else
                Utils::log("SENS", "SENSSimulator: Unable to open file: %s ", textFileName.c_str());
        }
    }
}
