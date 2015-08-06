#ifndef QTFRAMEGRABBER_H
#define QTFRAMEGRABBER_H

#include <QtMultiMedia/QAbstractVideoSurface>
#include <QCamera>
#include <QImage>
#include <atomic>

//-----------------------------------------------------------------------------
class qtFrameGrabber : public QAbstractVideoSurface
{
    Q_OBJECT
    public:
        explicit    qtFrameGrabber(QObject *parent = 0);
                   ~qtFrameGrabber();

        QList<QVideoFrame::PixelFormat> 
                    supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const;

        bool        present(const QVideoFrame &frame);

        void        stop();
        void        start();

        std::atomic<bool> readyToCopy;
        bool        isAvailable() {return _isAvailable;}
        void        copyToSLIfReady();
    
    private slots:
        void        updateCameraState(QCamera::State);
        void        displayCameraError();
        void        updateLockStatus(QCamera::LockStatus, QCamera::LockChangeReason);
private:
        QCamera*    _camera;
        QImage*     _lastImage;
        bool        _isAvailable;
        int         _glFormat;
};
//-----------------------------------------------------------------------------
#endif // QTFRAMEGRABBER_H

