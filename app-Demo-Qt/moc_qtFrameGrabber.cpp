/****************************************************************************
** Meta object code from reading C++ file 'qtFrameGrabber.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.5.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "include/qtFrameGrabber.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'qtFrameGrabber.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.5.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_qtFrameGrabber_t {
    QByteArrayData data[4];
    char stringdata0[37];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_qtFrameGrabber_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_qtFrameGrabber_t qt_meta_stringdata_qtFrameGrabber = {
    {
QT_MOC_LITERAL(0, 0, 14), // "qtFrameGrabber"
QT_MOC_LITERAL(1, 15, 14), // "frameAvailable"
QT_MOC_LITERAL(2, 30, 0), // ""
QT_MOC_LITERAL(3, 31, 5) // "frame"

    },
    "qtFrameGrabber\0frameAvailable\0\0frame"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_qtFrameGrabber[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   19,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::QImage,    3,

       0        // eod
};

void qtFrameGrabber::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        qtFrameGrabber *_t = static_cast<qtFrameGrabber *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->frameAvailable((*reinterpret_cast< QImage(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (qtFrameGrabber::*_t)(QImage );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&qtFrameGrabber::frameAvailable)) {
                *result = 0;
            }
        }
    }
}

const QMetaObject qtFrameGrabber::staticMetaObject = {
    { &QAbstractVideoSurface::staticMetaObject, qt_meta_stringdata_qtFrameGrabber.data,
      qt_meta_data_qtFrameGrabber,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *qtFrameGrabber::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *qtFrameGrabber::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_qtFrameGrabber.stringdata0))
        return static_cast<void*>(const_cast< qtFrameGrabber*>(this));
    return QAbstractVideoSurface::qt_metacast(_clname);
}

int qtFrameGrabber::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QAbstractVideoSurface::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void qtFrameGrabber::frameAvailable(QImage _t1)
{
    void *_a[] = { Q_NULLPTR, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
