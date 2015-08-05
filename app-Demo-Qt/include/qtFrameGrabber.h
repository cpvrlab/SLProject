#ifndef QTFRAMEGRABBER_H
#define QTFRAMEGRABBER_H

#include <QtMultiMedia/QAbstractVideoSurface>
#include <QCamera>
#include <QList>

#include <atomic>

//-----------------------------------------------------------------------------
class qtFrameGrabber : public QAbstractVideoSurface
{
Q_OBJECT
public:
    explicit        qtFrameGrabber(QObject *parent = 0);
                   ~qtFrameGrabber();

        QList<QVideoFrame::PixelFormat> 
                    supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const;

        bool        present(const QVideoFrame &frame);

        void        stop();
        void        start();

        std::atomic<bool> lastFrameIsConsumed;
        bool        isAvailable() {return _isAvailable;}

private:
        QCamera*    _camera;
        bool        _isAvailable;
};
//-----------------------------------------------------------------------------
#endif // QTFRAMEGRABBER_H

