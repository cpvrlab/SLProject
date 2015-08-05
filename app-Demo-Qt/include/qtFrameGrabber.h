#ifndef QTCAMERAFRAMEGRABBER_H
#define QTCAMERAFRAMEGRABBER_H

// Qt includes
#include <QtMultiMedia/QAbstractVideoSurface>
#include <QCamera>
#include <QImage>
#include <QList>

//-----------------------------------------------------------------------------
class qtFrameGrabber : public QAbstractVideoSurface
{
Q_OBJECT
public:
    explicit qtFrameGrabber(QObject *parent = 0);

    QList<QVideoFrame::PixelFormat> supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const;

    bool    present(const QVideoFrame &frame);

    void    stop();
    void    start();

    std::atomic<bool> lastFrameIsConsumed;

signals:
    void frameAvailable(QImage frame);

public slots:

private:
        QCamera     _camera;
        QImage      _lastFrame;
};
//-----------------------------------------------------------------------------
#endif // QTCAMERAFRAMEGRABBER_H

