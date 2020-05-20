#ifndef IMAGE_BUFFER_H
#define IMAGE_BUFFER_H

class ImageBuffer
{
public:
    void init(int numOfSlots, cv::Size imgSize)
    {
        _currentIdx = 0;
        _images.clear();
        //dont do this: _images.resize(numOfSlots, cv::Mat(imgSize.height, imgSize.width, CV_8UC3));
        _images.resize(numOfSlots);
        for (int i = 0; i < _images.size(); ++i)
            _images[i] = cv::Mat(imgSize.height, imgSize.width, CV_8UC3);
    }

    cv::Mat& outputSlot()
    {
        return _images[_currentIdx];
    }

    cv::Mat& inputSlot()
    {
        int inputIdx = _currentIdx - 1;
        if (inputIdx < 0)
            inputIdx = (int)_images.size() - 1;
        return _images[inputIdx];
    }

    void incrementSlot()
    {
        _currentIdx = (_currentIdx + 1) % _images.size();
    }

private:
    int                  _currentIdx = 0;
    std::vector<cv::Mat> _images;
};

#endif //IMAGE_BUFFER_H
