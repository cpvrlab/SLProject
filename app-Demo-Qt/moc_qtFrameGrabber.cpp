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
    QByteArrayData data[8];
    char stringdata0[131];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_qtFrameGrabber_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_qtFrameGrabber_t qt_meta_stringdata_qtFrameGrabber = {
    {
QT_MOC_LITERAL(0, 0, 14), // "qtFrameGrabber"
QT_MOC_LITERAL(1, 15, 17), // "updateCameraState"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 14), // "QCamera::State"
QT_MOC_LITERAL(4, 49, 18), // "displayCameraError"
QT_MOC_LITERAL(5, 68, 16), // "updateLockStatus"
QT_MOC_LITERAL(6, 85, 19), // "QCamera::LockStatus"
QT_MOC_LITERAL(7, 105, 25) // "QCamera::LockChangeReason"

    },
    "qtFrameGrabber\0updateCameraState\0\0"
    "QCamera::State\0displayCameraError\0"
    "updateLockStatus\0QCamera::LockStatus\0"
    "QCamera::LockChangeReason"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_qtFrameGrabber[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   29,    2, 0x08 /* Private */,
       4,    0,   32,    2, 0x08 /* Private */,
       5,    2,   33,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, 0x80000000 | 3,    2,
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 6, 0x80000000 | 7,    2,    2,

       0        // eod
};

void qtFrameGrabber::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        qtFrameGrabber *_t = static_cast<qtFrameGrabber *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateCameraState((*reinterpret_cast< QCamera::State(*)>(_a[1]))); break;
        case 1: _t->displayCameraError(); break;
        case 2: _t->updateLockStatus((*reinterpret_cast< QCamera::LockStatus(*)>(_a[1])),(*reinterpret_cast< QCamera::LockChangeReason(*)>(_a[2]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCamera::State >(); break;
            }
            break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCamera::LockChangeReason >(); break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QCamera::LockStatus >(); break;
            }
            break;
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
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 3)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
