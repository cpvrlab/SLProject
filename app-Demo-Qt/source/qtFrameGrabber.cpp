#include "qtFrameGrabber.h"
#include <QCameraInfo>
#include <QApplication>
#include <SLInterface.h>

Q_DECLARE_METATYPE(QCameraInfo)

//-----------------------------------------------------------------------------
qtFrameGrabber::qtFrameGrabber(QObject *parent) : QAbstractVideoSurface(parent)
{
    readyToCopy = false;
    _isAvailable = false;
    _camera = 0;
    _lastImage = 0;

    try
    {   
        QCameraInfo defaultCam = QCameraInfo::defaultCamera();
        _isAvailable = !defaultCam.isNull(); //QCameraInfo::availableCameras().count() > 0;
        if (_isAvailable)
        {   
            _camera = new QCamera(QCameraInfo::defaultCamera());

            connect(_camera, SIGNAL(stateChanged(QCamera::State)), 
                    this, SLOT(updateCameraState(QCamera::State)));
            connect(_camera, SIGNAL(error(QCamera::Error)), 
                    this, SLOT(displayCameraError()));
            connect(_camera, SIGNAL(lockStatusChanged(QCamera::LockStatus, QCamera::LockChangeReason)),
                    this, SLOT(updateLockStatus(QCamera::LockStatus, QCamera::LockChangeReason)));
            _camera->setViewfinder(this);
        }
    }
    catch (exception ex)
    {   cout << "qtFrameGrabber: Camera creation failed." << endl;
        _camera = 0;
    }
}
//-----------------------------------------------------------------------------
qtFrameGrabber::~qtFrameGrabber()
{
    delete _camera;
}
//-----------------------------------------------------------------------------
QList<QVideoFrame::PixelFormat> qtFrameGrabber::supportedPixelFormats(QAbstractVideoBuffer::HandleType handleType) const
{
    Q_UNUSED(handleType);
    return QList<QVideoFrame::PixelFormat>()
        << QVideoFrame::Format_ARGB32
        << QVideoFrame::Format_ARGB32_Premultiplied
        << QVideoFrame::Format_RGB32
        << QVideoFrame::Format_RGB24
        << QVideoFrame::Format_RGB565
        << QVideoFrame::Format_RGB555
        << QVideoFrame::Format_ARGB8565_Premultiplied
        << QVideoFrame::Format_BGRA32
        << QVideoFrame::Format_BGRA32_Premultiplied
        << QVideoFrame::Format_BGR32
        << QVideoFrame::Format_BGR24
        << QVideoFrame::Format_BGR565
        << QVideoFrame::Format_BGR555
        << QVideoFrame::Format_BGRA5658_Premultiplied
        << QVideoFrame::Format_AYUV444
        << QVideoFrame::Format_AYUV444_Premultiplied
        << QVideoFrame::Format_YUV444
        << QVideoFrame::Format_YUV420P
        << QVideoFrame::Format_YV12
        << QVideoFrame::Format_UYVY
        << QVideoFrame::Format_YUYV
        << QVideoFrame::Format_NV12
        << QVideoFrame::Format_NV21
        << QVideoFrame::Format_IMC1
        << QVideoFrame::Format_IMC2
        << QVideoFrame::Format_IMC3
        << QVideoFrame::Format_IMC4
        << QVideoFrame::Format_Y8
        << QVideoFrame::Format_Y16
        << QVideoFrame::Format_Jpeg
        << QVideoFrame::Format_CameraRaw
        << QVideoFrame::Format_AdobeDng;
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::start()
{
    if (!_camera) return;

    try
    {
        QCamera::State currentState = _camera->state();
        if (currentState != QCamera::ActiveState)
        {   _camera->start();
            QApplication::processEvents();
        }
    } 
    catch(exception ex)
    {
        cout << "Camera Start failed" << endl;
    }
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::stop()
{
    if (!_camera) return;
    QCamera::State currentState = _camera->state();
    if (currentState == QCamera::ActiveState)
    {   _camera->stop();
        QApplication::processEvents();
    }
}
//-----------------------------------------------------------------------------
bool qtFrameGrabber::present(const QVideoFrame &frame)
{
    bool frameIsValid = frame.isValid();
    if (_camera && slNeedsVideoImage() && frameIsValid && !readyToCopy)
    {
        QVideoFrame cloneFrame(frame);
        cloneFrame.map(QAbstractVideoBuffer::ReadOnly);

        delete _lastImage;

        _lastImage = new QImage (cloneFrame.bits(),
                                 cloneFrame.width(),
                                 cloneFrame.height(),
                                 QVideoFrame::imageFormatFromPixelFormat(cloneFrame.pixelFormat()));

        // Set the according OpenGL format
        switch(cloneFrame.pixelFormat())
        {   case QVideoFrame::Format_Y8:     _glFormat = GL_LUMINANCE; break;
            case QVideoFrame::Format_BGR24:  _glFormat = GL_RGB; break;
            case QVideoFrame::Format_RGB24:  _glFormat = GL_RGB; break;
            case QVideoFrame::Format_ARGB32: _glFormat = GL_RGBA; break;
            case QVideoFrame::Format_BGRA32: _glFormat = GL_RGBA; break;
            default:
                SL_EXIT_MSG("qtFrameGrabber::present: Qt image format not supported");
                cloneFrame.unmap();
                return false;
        }

        SLuchar* p = cloneFrame.bits();

        cloneFrame.unmap();

        readyToCopy = true;
        return true;
    }
    return false;
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::copyToSLIfReady()
{
    start();
    if (readyToCopy)
    {
        slCopyVideoImage(_lastImage->width(),
                         _lastImage->height(),
                         _glFormat,
                         _lastImage->bits(),
                         true);
        readyToCopy = false;
    }
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::updateCameraState(QCamera::State state)
{
    switch (state) 
    {
        case QCamera::ActiveState:   
            cout << "QCamera::ActiveState" << endl; break;
        case QCamera::UnloadedState:
            cout << "QCamera::UnloadedState" << endl; break;
        case QCamera::LoadedState:
            cout << "QCamera::LoadedState" << endl; break;
    }
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::displayCameraError()
{
    cout << _camera->errorString().toStdString() << endl;
}
//-----------------------------------------------------------------------------
void qtFrameGrabber::updateLockStatus(QCamera::LockStatus status, QCamera::LockChangeReason reason)
{
    switch (status) 
    {
    case QCamera::Searching:
        cout << "QCamera::Searching" << endl;
        break;
    case QCamera::Locked:
        cout << "QCamera::Locked" << endl;
        break;
    case QCamera::Unlocked:
        cout << "QCamera::Unlocked" << endl;
    }
}
//-----------------------------------------------------------------------------
