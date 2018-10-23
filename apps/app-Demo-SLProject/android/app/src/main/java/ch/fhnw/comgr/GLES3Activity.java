//#############################################################################
//  File:      GLES3Activity.java
//  Author:    Marcus Hudritsch, Zingg Pascal
//  Date:      Spring 2017
//  Purpose:   Android Java toplevel activity class
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Zingg Pascal
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Please do not change the name space. The SLProject app is identified in the app-store with it.
package ch.fhnw.comgr;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraCharacteristics;
import android.location.Location;
import android.location.LocationManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.support.annotation.NonNull;

import java.io.IOException;


public class GLES3Activity extends Activity implements View.OnTouchListener, SensorEventListener {
    GLES3View                   myView;             // OpenGL view
    static int                  pointersDown = 0;   // NO. of fingers down
    static long                 lastTouchMS = 0;    // Time of last touch in ms

    private static final String TAG = "SLProject";
    private static final int PERMISSIONS_MULTIPLE_REQUEST = 123;

    private int                     _currentVideoType;
    private boolean                 _cameraPermissionGranted;
    private boolean                 _permissionRequestIsOpen;
    private boolean                 _rotationSensorIsRunning = false;
    private long                    _rotationSensorStartTime = 0; //Time when rotation sensor was started
    private boolean                 _locationPermissionGranted;
    private boolean                 _locationSensorIsRunning = false;
    private LocationManager         _locationManager;
    private GeneralLocationListener _locationListener;

    @Override
    protected void onCreate(Bundle icicle) {
        Log.i(TAG, "GLES3Activity.onCreate");
        super.onCreate(icicle);

        // Extract (unzip) files in APK
        try {
            Log.i(TAG, "extractAPK");
            GLES3Lib.App = getApplication();
            GLES3Lib.extractAPK();
        } catch (IOException e) {
            Log.e(TAG, "Error extracting files from the APK archive: " + e.getMessage());
        }

        // Create view
        myView = new GLES3View(GLES3Lib.App);
        GLES3Lib.view = myView;
        GLES3Lib.activity = this;
        myView.setOnTouchListener(this);
        Log.i(TAG, "setContentView");
        setContentView(myView);

        // Get display resolution. This is used to scale the menu buttons accordingly
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);
        int dpi = (int) (((float) metrics.xdpi + (float) metrics.ydpi) * 0.5);
        GLES3Lib.dpi = dpi;
        Log.i(TAG, "DisplayMetrics: " + dpi);

        // Init Camera (the camera is started by cameraStart from within the view renderer)
        Log.i(TAG, "Request camera permission ...");
        //If we are on android 5.1 or lower the permission was granted during installation.
        //On Android 6 or higher it requests a dangerous permission during runtime.
        //On Android 7 there could be problems that permissions where not granted
        //(Huawei Honor 8 must enable soecial log setting by dialing *#*#2846579#*#*)

        // Check permissions all at once (from Android M onwards)
        Log.i(TAG, "Request Camera and GPS permission ...");
        if (    ActivityCompat.checkSelfPermission(GLES3Activity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(GLES3Activity.this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(GLES3Activity.this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            _cameraPermissionGranted = true;
            _locationPermissionGranted = true;
        }
        else {
            _permissionRequestIsOpen = true;
            ActivityCompat.requestPermissions(GLES3Activity.this, new String[]{
                    Manifest.permission.CAMERA,
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.ACCESS_FINE_LOCATION}, PERMISSIONS_MULTIPLE_REQUEST);
        }
    }

    // After on onCreate
    @Override
    protected void onStart() {
        Log.i(TAG, "GLES3Activity.onStart");
        super.onStart();
    }

    // Another activity comes into foreground but this is still visible (e.g. with the home button)
    @Override
    protected void onPause() {
        Log.i(TAG, "GLES3Activity.onPause");
        super.onPause();
    }

    @Override
    // My activity is no longer visible
    protected void onStop() {
        Log.i(TAG, "GLES3Activity.onStop");

        // Stop sensors to save energy
        cameraStop();
        locationSensorStop();
        rotationSensorStop();

        super.onStop();
    }

    // The user resumed this activity
    @Override
    protected void onResume() {
        Log.i(TAG, "GLES3Activity.onResume");
        super.onResume();
    }

    // A stopped but not destroyed activity is reactivated
    @Override
    protected void onRestart() {
        Log.i(TAG, "GLES3Activity.onRestart");
        super.onRestart();
    }

    @Override
    // The process of this activity is getting killed (e.g. with the back button)
    protected void onDestroy() {
        Log.i(TAG, "GLES3Activity.onDestroy");
        myView.queueEvent(new Runnable() {public void run() {GLES3Lib.onClose();}});
        super.onDestroy();
        finish();
    }

    @Override
    public boolean onTouch(View v, final MotionEvent event) {
        if (event == null) {
            Log.i(TAG, "onTouch: null event");
            return false;
        }

        int action = event.getAction();
        int actionCode = action & MotionEvent.ACTION_MASK;

        try {
            if (actionCode == MotionEvent.ACTION_DOWN ||
                    actionCode == MotionEvent.ACTION_POINTER_DOWN)
                return handleTouchDown(event);
            else if (actionCode == MotionEvent.ACTION_UP ||
                    actionCode == MotionEvent.ACTION_POINTER_UP)
                return handleTouchUp(event);
            else if (actionCode == MotionEvent.ACTION_MOVE)
                return handleTouchMove(event);
            else Log.i(TAG, "Unhandeled Event: " + actionCode);
        } catch (Exception ex) {
            Log.i(TAG, "onTouch (Exception: " + actionCode);
        }

        return false;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.i(TAG, String.format("onAccuracyChanged"));
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR && _rotationSensorIsRunning) {

            //let some time pass until we process these values
            if (System.currentTimeMillis() - _rotationSensorStartTime < 500 )
                return;

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
            SensorManager.getQuaternionFromVector(Q, event.values);

            // Send the quaternion as x,y,z & w to SLScene::onRotationQUAT
            // See the following routines how the rotation is used:
            // SLScene::onRotationQUAT calculates the offset if _zeroYawAtStart is true
            // SLCamera::setView how the device rotation is processed for the camera's view
            myView.queueEvent(new Runnable() {public void run() {GLES3Lib.onRotationQUAT(Q[1],Q[2],Q[3],Q[0]);}});
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String permissions[], @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_MULTIPLE_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "onRequestPermissionsResult: CAMERA permission granted.");
                _cameraPermissionGranted = true;
            } else {
                Log.i(TAG, "onRequestPermissionsResult: CAMERA permission refused.");
                _cameraPermissionGranted = false;
            }
            if (grantResults.length > 2 &&
                    grantResults[1] == PackageManager.PERMISSION_GRANTED &&
                    grantResults[2] == PackageManager.PERMISSION_GRANTED) {
                Log.i(TAG, "onRequestPermissionsResult: GPS sensor permission granted.");
                _locationPermissionGranted = true;
            } else {
                Log.i(TAG, "onRequestPermissionsResult: GPS sensor permission refused.");
                _locationPermissionGranted = false;
            }
            _permissionRequestIsOpen = false;
        }
    }


    /**
     * Events:
     * <p>
     * Finger Down
     * -----------
     * Just tap on screen -> onMouseDown, onMouseUp
     * Tap and hold       -> onMouseDown
     * release   -> onMouseUp
     * 2 Fingers same time -> onTouch2Down
     * 2 Fingers not same time -> onMouseDown, onMouseUp, onTouch2Down
     * <p>
     * Finger Up
     * ---------
     * 2 Down, release one -> onTouch2Up
     * release other one -> onMouseUp
     * 2 Down, release one, put another one down -> onTouch2Up, onTouch2Down
     * 2 Down, release both same time -> onTouch2Up
     * 2 Down, release both not same time -> onTouch2Up
     */

    public boolean handleTouchDown(final MotionEvent event) {
        int touchCount = event.getPointerCount();
        final int x0 = (int) event.getX(0);
        final int y0 = (int) event.getY(0);
        //Log.i(TAG, "Dn:" + touchCount);

        // just got a new single touch
        if (touchCount == 1) {
        
            // get time to detect double taps
            long touchNowMS = System.currentTimeMillis();
            long touchDeltaMS = touchNowMS - lastTouchMS;
            lastTouchMS = touchNowMS;

            if (touchDeltaMS < 250)
                myView.queueEvent(new Runnable() {public void run() {
                    GLES3Lib.onDoubleClick(1, x0, y0);
                }});
            else
                myView.queueEvent(new Runnable() {public void run() {
                    GLES3Lib.onMouseDown(1, x0, y0);
                }});
        }

        // it's two fingers but one delayed (already executed mouse down
        else if (touchCount == 2 && pointersDown == 1) {
            final int x1 = (int) event.getX(1);
            final int y1 = (int) event.getY(1);
            myView.queueEvent(new Runnable() {public void run() {
                    GLES3Lib.onMouseUp(1, x0, y0);
                }});
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onTouch2Down(x0, y0, x1, y1);
                }
            });
            // it's two fingers at the same time
        } else if (touchCount == 2) {
            // get time to detect double taps
            long touchNowMS = System.currentTimeMillis();
            long touchDeltaMS = touchNowMS - lastTouchMS;
            lastTouchMS = touchNowMS;

            final int x1 = (int) event.getX(1);
            final int y1 = (int) event.getY(1);

            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onTouch2Down(x0, y0, x1, y1);
                }
            });
        }
        pointersDown = touchCount;
        myView.requestRender();
        return true;
    }

    public boolean handleTouchUp(final MotionEvent event) {
        int touchCount = event.getPointerCount();
        //Log.i(TAG, "Up:" + touchCount + " x: " + (int)event.getX(0) + " y: " + (int)event.getY(0));
        final int x0 = (int) event.getX(0);
        final int y0 = (int) event.getY(0);
        if (touchCount == 1) {
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onMouseUp(1, x0, y0);
                }
            });
        } else if (touchCount == 2) {
            final int x1 = (int) event.getX(1);
            final int y1 = (int) event.getY(1);
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onTouch2Up(x0, y0, x1, y1);
                }
            });
        }

        pointersDown = touchCount;
        myView.requestRender();
        return true;
    }

    public boolean handleTouchMove(final MotionEvent event) {
        final int x0 = (int) event.getX(0);
        final int y0 = (int) event.getY(0);
        int touchCount = event.getPointerCount();
        //Log.i(TAG, "Mv:" + touchCount);

        if (touchCount == 1) {
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onMouseMove(x0, y0);
                }
            });
        } else if (touchCount == 2) {
            final int x1 = (int) event.getX(1);
            final int y1 = (int) event.getY(1);
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onTouch2Move(x0, y0, x1, y1);
                }
            });
        }
        myView.requestRender();
        return true;
    }

    /**
     * Starts the camera service if not running.
     * It is called from the GL view renderer thread.
     * While the service is starting no other calls to startService are allowed.
     *
     * @param requestedVideoType (0 = GLES3Lib.VIDEO_TYPE_NONE, 1 = *_MAIN, 2 = *_SCND)
     * @param requestedVideoSizeIndex (0 = 640x480, -1 = the next smaller, +1 = the next bigger)
     */
    public void cameraStart(int requestedVideoType, int requestedVideoSizeIndex) {
        if (!_cameraPermissionGranted) return;

        if (!GLES3Camera2Service.isTransitioning) {
            if (!GLES3Camera2Service.isRunning) {
                GLES3Camera2Service.isTransitioning = true;
                GLES3Camera2Service.requestedVideoSizeIndex = requestedVideoSizeIndex;

                if (requestedVideoType == GLES3Lib.VIDEO_TYPE_MAIN) {
                    GLES3Camera2Service.videoType = CameraCharacteristics.LENS_FACING_BACK;
                    Log.i(TAG, "Going to start main back camera service ...");
                } else {
                    GLES3Camera2Service.videoType = CameraCharacteristics.LENS_FACING_FRONT;
                    Log.i(TAG, "Going to start front camera service ...");
                }

                //////////////////////////////////////////////////////////////////////
                startService(new Intent(getBaseContext(), GLES3Camera2Service.class));
                //////////////////////////////////////////////////////////////////////

                _currentVideoType = requestedVideoType;
            } else {
                // if the camera is running the type or size is different we first stop the camera
                if (requestedVideoType != _currentVideoType ||
                    requestedVideoSizeIndex != GLES3Camera2Service.requestedVideoSizeIndex) {
                    GLES3Camera2Service.isTransitioning = true;
                    Log.i(TAG, "Going to stop camera service to change type ...");
                    stopService(new Intent(getBaseContext(), GLES3Camera2Service.class));
                }
            }
        }
    }

    /**
     * Stops the camera service if running.
     * It is called from the GL view renderer thread.
     * While the service is stopping no other calls to stopService are allowed.
     */
    public void cameraStop() {
        if (!_cameraPermissionGranted) return;

        if (!GLES3Camera2Service.isTransitioning) {
            if (GLES3Camera2Service.isRunning) {
                GLES3Camera2Service.isTransitioning = true;
                Log.i(TAG, "Going to stop camera service ...");

                /////////////////////////////////////////////////////////////////////
                stopService(new Intent(getBaseContext(), GLES3Camera2Service.class));
                /////////////////////////////////////////////////////////////////////
            }
        }
    }

    /**
     * Registers the the rotation sensor listener
     * It is called from the GL view renderer thread.
     */
    public void rotationSensorStart()
    {
        if (_rotationSensorIsRunning)
            return;

        // Init Sensor
        try {
            SensorManager sm = (SensorManager) getSystemService(SENSOR_SERVICE);
            if (sm != null) {
                sm.registerListener(this,
                        sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR),
                        sm.SENSOR_DELAY_GAME);
                _rotationSensorStartTime = System.currentTimeMillis();
                _rotationSensorIsRunning = true;
            } else {
                _rotationSensorIsRunning = true;
            }
        }
        catch (Exception e) {
            Log.i(TAG, "Exception: " + e.getMessage());
            _rotationSensorIsRunning = false;
        }
        Log.d(TAG, "Rotation Sensor is running: "+ _rotationSensorIsRunning);
    }

    /**
     * Unregisters the the rotation sensor listener
     * It is called from the GL view renderer thread.
     */
    public void rotationSensorStop()
    {
        if (!_rotationSensorIsRunning)
            return;

        // Init Sensor
        try {
            SensorManager sm = (SensorManager) getSystemService(SENSOR_SERVICE);
            if (sm != null) {
                sm.unregisterListener(this, sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR));
            }
            _rotationSensorIsRunning = false;
        }
        catch (Exception e) {
            Log.i(TAG, "Exception: " + e.getMessage());
            _rotationSensorIsRunning = false;
        }
        Log.d(TAG, "Rotation Sensor is running: "+ _rotationSensorIsRunning);
    }

    /**
     * Starts the location manager.
     */
    @SuppressWarnings("ResourceType")
    public void locationSensorStart() {
        // Create GPS manager and listener
        if (_locationSensorIsRunning)
            return;

        if (_locationListener == null) {
            _locationListener = new GeneralLocationListener(this, "GPS");
        }

        _locationManager = (LocationManager) getSystemService(Context.LOCATION_SERVICE);

        if (_locationManager != null && _locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            Log.i(TAG, "Requesting GPS location updates");
            _locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                                                    1000,
                                                    0,
                                                    _locationListener);
            _locationSensorIsRunning = true;
        } else {
            _locationSensorIsRunning = false;
        }
        Log.d(TAG, "GPS Sensor is running: "+ _locationSensorIsRunning);
    }

    /**
     * Stops the location managers
     */
    @SuppressWarnings("ResourceType")
    public void locationSensorStop() {
        if (_locationListener != null) {
            Log.d(TAG, "Removing _locationManager updates");
            _locationManager.removeUpdates(_locationListener);
            _locationListener = null;
        }
    }

    /**
     * Stops location manager, then starts it.
     */
    public void locationSensorRestart() {
        Log.d(TAG, "Restarting location managers");
        locationSensorStop();
        locationSensorStart();
    }

    /**
     * This event is raised when the GeneralLocationListener has a new location.
     * This method in turn updates notification, writes to file, reobtains
     * preferences, notifies main service client and resets location managers.
     *
     * @param loc Location object
     */
    public void onLocationChanged(Location loc) {
        //long currentTimeStamp = System.currentTimeMillis();
        //if (!loc.hasAccuracy() || loc.getAccuracy() == 0) return;

        Log.i(TAG, "onLocationChanged: " + String.valueOf(loc.getLatitude()) + "," + String.valueOf(loc.getLongitude()));
        myView.queueEvent(new Runnable() {
            public void run() {
                GLES3Lib.onLocationLLA(
                        loc.getLatitude(),
                        loc.getLongitude(),
                        loc.getAltitude(),
                        loc.getAccuracy());
            }
        });
    }
}
