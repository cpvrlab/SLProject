#ifndef APP_WAI_SLAM_PARAM_HELPER_H
#define APP_WAI_SLAM_PARAM_HELPER_H

#include <Utils.h>
#include <string>

struct SlamVideoInfos
{
    std::string dateTime;
    std::string weatherConditions;
    std::string deviceString;
    std::string purpose;
    std::string resolution;
};

struct SlamMapInfos
{
    std::string dateTime;
    std::string location;
    std::string area;
    std::string extractorType;
    int         nLevels;
};

static bool extractSlamVideoInfosFromFileName(std::string     fileName,
                                              SlamVideoInfos* slamVideoInfos)
{
    bool result = false;

    std::vector<std::string> stringParts;
    Utils::splitString(fileName, '_', stringParts);

    if (stringParts.size() == 5)
    {
        slamVideoInfos->dateTime          = stringParts[0];
        slamVideoInfos->weatherConditions = stringParts[1];
        slamVideoInfos->deviceString      = stringParts[2];
        slamVideoInfos->purpose           = stringParts[3];
        slamVideoInfos->resolution        = stringParts[4];

        result = true;
    }

    return result;
}

static std::string constructSlamMapIdentifierString(std::string location,
                                                    std::string area,
                                                    std::string extractorType,
                                                    int         nLevels,
                                                    std::string dateTime = "")
{
    if (dateTime.empty())
    {
        dateTime = Utils::getDateTime2String();
    }

    std::string result = "DEVELOPMENT-map_" + dateTime + "_" + location + "_" + area + "_" + extractorType + "_" + std::to_string(nLevels);

    return result;
}

static bool extractSlamMapInfosFromFileName(std::string   fileName,
                                            SlamMapInfos* slamMapInfos)
{
    bool result = false;

    *slamMapInfos = {};

    fileName = Utils::getFileNameWOExt(fileName);

    std::vector<std::string> stringParts;
    Utils::splitString(fileName, '_', stringParts);

    if (stringParts.size() >= 4)
    {
        slamMapInfos->dateTime = stringParts[1];
        slamMapInfos->location = stringParts[2];
        slamMapInfos->area     = stringParts[3];

        result = true;
    }

    if (stringParts.size() >= 5)
    {
        slamMapInfos->extractorType = stringParts[4];
    }

    if (stringParts.size() >= 6)
    {
        slamMapInfos->nLevels = std::stoi(stringParts[5]);
    }
    return result;
}

static std::string constructSlamLocationDir(std::string locationsRootDir, std::string location)
{
    std::string result = Utils::unifySlashes(locationsRootDir) + location + "/";

    return result;
}

static std::string constructSlamAreaDir(std::string locationsRootDir, std::string location, std::string area)
{
    std::string result = constructSlamLocationDir(locationsRootDir, location) + Utils::getFileNameWOExt(area) + "/";

    return result;
}

static std::string constructSlamMapDir(std::string locationsRootDir, std::string location, std::string area)
{
    std::string result = constructSlamAreaDir(locationsRootDir, location, area) + "maps/";

    return result;
}

static std::string constructSlamMapImgDir(std::string mapDir, std::string mapFileName)
{
    SlamMapInfos mapInfos;

    std::string result = Utils::unifySlashes(mapDir + Utils::getFileNameWOExt(mapFileName));

    return result;
}

static std::string constructSlamVideoDir(std::string locationsRootDir, std::string location, std::string area)
{
    std::string result = constructSlamAreaDir(locationsRootDir, location, area) + "videos/";

    return result;
}

static std::string constructSlamMapFileName(std::string location,
                                            std::string area,
                                            std::string extractorType,
                                            int         nLevels,
                                            std::string dateTime  = "",
                                            bool        binaryMap = false)
{
    std::string result = constructSlamMapIdentifierString(location, area, extractorType, nLevels, dateTime);
    if (binaryMap)
        result += ".waimap";
    else
        result += ".json.gz";
    return result;
}

static std::string constructBinarySlamMapFileName(std::string location,
                                                  std::string area,
                                                  std::string extractorType,
                                                  int         nLevels,
                                                  std::string dateTime = "")
{
    std::string result = constructSlamMapIdentifierString(location, area, extractorType, nLevels, dateTime) + ".waimap";

    return result;
}

static std::string constructSlamMarkerDir(std::string locationsRootDir, std::string location, std::string area)
{
    std::string result = constructSlamAreaDir(locationsRootDir, location, area) + "markers/";

    return result;
}

#endif
