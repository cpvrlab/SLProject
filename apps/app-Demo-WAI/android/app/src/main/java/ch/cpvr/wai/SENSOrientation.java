package ch.cpvr.wai;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import static android.content.Context.SENSOR_SERVICE;

//java sensor backend for SENSNdkOrientation cpp class
public class SENSOrientation {

    Context _context;
    SensorManager _sensorManager;
    SENSOrientationListener _sensorListener;
    boolean _isRunning = false;

    //set java activity context
    public void init(Context context) {
        _context = context;
    }

    public void start() {
        Log.i("SENSOrientation", "start()");

        if(_context == null) {
            Log.i("SENSOrientation", "start: you have to call init first!");
            return;
        }

        if(_isRunning)
            return;

        if(_sensorManager == null) {
            _sensorManager = (SensorManager) _context.getSystemService(SENSOR_SERVICE);
        }

        if(_sensorListener == null) {
            _sensorListener = new SENSOrientationListener(_context);
        }

        if (_sensorManager != null) {
            Log.i("SENSOrientation", "Requesting GPS location updates");

            _isRunning = true;
            Activity activity = (Activity)_context;
            activity.runOnUiThread(new Runnable() {
                public void run() {
                    _sensorManager.registerListener(_sensorListener,
                            _sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR),
                            _sensorManager.SENSOR_DELAY_GAME);
                }
            });

        } else {
            _isRunning = false;
        }
    }

    public void stop() {
        if(!_isRunning)
            return;

        _isRunning = false;
    }

    native static void onOrientationQuat(float quatX, float quatY, float quatZ, float quatW);
}


class SENSOrientationListener implements SensorEventListener {

    private static final String TAG = "SENSGps";
    Context _context;

    public SENSOrientationListener(Context context) {
        _context = context;
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        if (sensorEvent.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {

            //let some time pass until we process these values
            //if (System.currentTimeMillis() - _rotationSensorStartTime < 500 )
            //    return;

            /*
            // Get 3x3 rotation matrix from XYZ-rotation vector (see docs)
            float R[] = new float[9];
            SensorManager.getRotationMatrixFromVector(R, event.values);

            // Get yaw, pitch & roll rotation angles in radians from rotation matrix
            float[] YPR = new float[3];
            SensorManager.getOrientation(R, YPR);

            // Send the euler angles as pitch, yaw & roll to SLScene::onRotationPYR
            final float y = YPR[0];
            final float p = YPR[1];
            final float r = YPR[2];
            myView.queueEvent(new Runnable() {public void run() {GLES3Lib.onRotationPYR(p, y, r);}});
            */

            // Get the rotation quaternion from the XYZ-rotation vector (see docs)
            final float Q[] = new float[4];
            SensorManager.getQuaternionFromVector(Q, sensorEvent.values);

            // Send the quaternion as x,y,z & w
            SENSOrientation.onOrientationQuat(Q[1],Q[2],Q[3],Q[0]);
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }
}
