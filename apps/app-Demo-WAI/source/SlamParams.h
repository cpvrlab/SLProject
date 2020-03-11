#ifndef SLAMPARAMS_H
#define SLAMPARAMS_H

struct ExtractorIds
{
    ExtractorType trackingExtractorId;
    ExtractorType initializationExtractorId;
    ExtractorType markerExtractorId;
};

struct SlamParams
{
    //returns true if loading was successful. Otherwise there may have been no file.
    bool load(std::string fileName)
    {
        cv::FileStorage fs;
        try
        {
            fs.open(fileName, cv::FileStorage::READ);
            if (fs.isOpened())
            {
                if (!fs["videoFile"].empty())
                    fs["videoFile"] >> videoFile;
                if (!fs["mapFile"].empty())
                    fs["mapFile"] >> mapFile;
                if (!fs["calibrationFile"].empty())
                    fs["calibrationFile"] >> calibrationFile;
                if (!fs["vocabularyFile"].empty())
                    fs["vocabularyFile"] >> vocabularyFile;
                if (!fs["markerFile"].empty())
                    fs["markerFile"] >> markerFile;
                if (!fs["location"].empty())
                    fs["location"] >> location;
                if (!fs["area"].empty())
                    fs["area"] >> area;

                if (!fs["cullRedundantPerc"].empty())
                    fs["cullRedundantPerc"] >> params.cullRedundantPerc;
                if (!fs["fixOldKfs"].empty())
                    fs["fixOldKfs"] >> params.fixOldKfs;
                if (!fs["onlyTracking"].empty())
                    fs["onlyTracking"] >> params.onlyTracking;
                if (!fs["retainImg"].empty())
                    fs["retainImg"] >> params.retainImg;
                if (!fs["serial"].empty())
                    fs["serial"] >> params.serial;
                if (!fs["trackOptFlow"].empty())
                    fs["trackOptFlow"] >> params.trackOptFlow;

                if (!fs["initializationExtractorId"].empty())
                    fs["initializationExtractorId"] >> extractorIds.initializationExtractorId;
                if (!fs["markerExtractorId"].empty())
                    fs["markerExtractorId"] >> extractorIds.markerExtractorId;
                if (!fs["trackingExtractorId"].empty())
                    fs["trackingExtractorId"] >> extractorIds.trackingExtractorId;

                fs.release();
                Utils::log("WAIApp", "SlamParams loaded from %s", fileName.c_str());

                return true;
            }
        }
        catch (...)
        {
            Utils::log("WAIApp", "SlamParams: Parsing of file failed: %s", fileName.c_str());
        }

        return false;
    }

    void save(std::string fileName)
    {
        cv::FileStorage fs(fileName, cv::FileStorage::WRITE);

        if (!fs.isOpened())
        {
            Utils::log("WAIApp", "SlamParams: Failed to open file for writing: %s", fileName.c_str());
            return;
        }

        fs << "videoFile" << videoFile;
        fs << "mapFile" << mapFile;
        fs << "calibrationFile" << calibrationFile;
        fs << "vocabularyFile" << vocabularyFile;
        fs << "markerFile" << markerFile;
        fs << "location" << location;
        fs << "area" << area;

        fs << "cullRedundantPerc" << params.cullRedundantPerc;
        fs << "fixOldKfs" << params.fixOldKfs;
        fs << "onlyTracking" << params.onlyTracking;
        fs << "retainImg" << params.retainImg;
        fs << "serial" << params.serial;
        fs << "trackOptFlow" << params.trackOptFlow;

        fs << "initializationExtractorId" << extractorIds.initializationExtractorId;
        fs << "markerExtractorId" << extractorIds.markerExtractorId;
        fs << "trackingExtractorId" << extractorIds.trackingExtractorId;

        fs.release();
        Utils::log("WAIApp", "SlamParams saved to %s", fileName.c_str());
    }

    std::string               videoFile;
    std::string               mapFile;
    std::string               calibrationFile;
    std::string               vocabularyFile;
    std::string               markerFile;
    std::string               location;
    std::string               area;
    WAI::ModeOrbSlam2::Params params;
    ExtractorIds              extractorIds;
};

#endif //SLAMPARAMS_H
