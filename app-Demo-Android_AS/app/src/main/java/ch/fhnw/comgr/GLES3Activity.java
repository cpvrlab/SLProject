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

package ch.fhnw.comgr;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import java.io.IOException;


public class GLES3Activity extends Activity implements View.OnTouchListener, SensorEventListener
{
    GLES3View myView;               // OpenGL view
    static int pointersDown = 0;    // NO. of fingers down
    static long lastTouchMS = 0;    // Time of last touch in ms
    private SensorManager mSensorManager;

    private static final String TAG = "SLProject";


    @Override
    protected void onCreate(Bundle icicle)
    {
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
        myView.setOnTouchListener(this);
        Log.i(TAG, "setContentView");
        setContentView(myView);

        // Get display resolution. This is used to scale the menu buttons accordingly
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);
        int dpi = (int) (((float) metrics.xdpi + (float) metrics.ydpi) * 0.5);
        GLES3Lib.dpi = dpi;
        Log.i(TAG, "DisplayMetrics: " + dpi);

        // Init Sensor
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        // Init Camera
        Log.i(TAG, "Request camera permission ...");
        ActivityCompat.requestPermissions(GLES3Activity.this, new String[]{Manifest.permission.CAMERA}, 1);

        Log.i(TAG, "Going to start camera service...");
        startService(new Intent(getBaseContext(), GLES3Camera2Service.class));

    }

    @Override
    protected void onPause()
    {
        Log.i(TAG, "GLES3Activity.onPause");
        super.onPause();
        myView.onPause();
    }

    @Override
    protected void onResume()
    {
        Log.i(TAG, "GLES3Activity.onResume");
        super.onResume();
        myView.onResume();

        if (mSensorManager != null)
            mSensorManager.registerListener(this,
                                            mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                                            SensorManager.SENSOR_DELAY_FASTEST);
    }

    @Override
    protected void onDestroy()
    {
        Log.i(TAG, "GLES3Activity.onDestroy");
        super.onDestroy();
    }

    @Override
    public boolean onTouch(View v, final MotionEvent event)
    {
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
    public boolean onCreateOptionsMenu(Menu menu)
    {
        Log.i(TAG, "onCreateOptionsMenu");
        myView.queueEvent(new Runnable() {
            public void run() {
                GLES3Lib.onMenuButton();
            }
        });
        return false;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy)
    {
        Log.i(TAG, String.format("onAccuracyChanged"));
    }

    @Override
    public void onSensorChanged(SensorEvent event)
    {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR && GLES3Lib.usesRotation()) {
            // The ROTATION_VECTOR sensor is a virtual fusion sensor
            // The quality strongly depends on the underlying algorithm and on
            // the sensor manufacturer. (See also chapter 7 in the book:
            // "Professional Sensor Programming (WROX Publishing)"

            // Get 3x3 rotation matrix from XYZ-rotation vector (see docs)
            float R[] = new float[9];
            SensorManager.getRotationMatrixFromVector(R, event.values);

            // Get yaw, pitch & roll rotation angles in radians from rotation matrix
            float[] YPR = new float[3];
            SensorManager.getOrientation(R, YPR);

            // Check display orientation (a preset orientation is set in the AndroidManifext.xml)
            Display display = getWindowManager().getDefaultDisplay();
            DisplayMetrics displaymetrics = new DisplayMetrics();
            display.getMetrics(displaymetrics);
            int screenWidth = displaymetrics.widthPixels;
            int screenHeight = displaymetrics.heightPixels;

            if (screenWidth < screenHeight) {    // Map pitch, yaw and roll to portrait display orientation
                final float p = YPR[1] * -1.0f - (float) Math.PI * 0.5f;
                final float y = YPR[0] * -1.0f;
                final float r = YPR[2] * -1.0f;
                myView.queueEvent(new Runnable() {
                    public void run() {
                        GLES3Lib.onRotationPYR(p, y, r);
                    }
                });
            } else {    // Map pitch, yaw and roll to landscape display orientation for Oculus Rift conformance
                final float p = YPR[2] * -1.0f - (float) Math.PI * 0.5f;
                final float y = YPR[0] * -1.0f;
                final float r = YPR[1];
                myView.queueEvent(new Runnable() {
                    public void run() {
                        GLES3Lib.onRotationPYR(p, y, r);
                    }
                });
            }

			/*
            // Get the rotation quaternion from the XYZ-rotation vector (see docs)
			final float Q[] = new float[4];
			SensorManager.getQuaternionFromVector(Q, event.values);
			myView.queueEvent(new Runnable() {public void run() {GLES3Lib.onRotationQUAT(Q[1],Q[2],Q[3],Q[0]);}});
			*/
        }
    }




    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults)
    {
        switch (requestCode)
        {
            case 1: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.
                } else {
                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                    Toast.makeText(GLES3Activity.this, "Permission denied ", Toast.LENGTH_SHORT).show();
                }
                return;
            }
            // other 'case' lines to check for other
            // permissions this app might request
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

    public boolean handleTouchDown(final MotionEvent event)
    {
        int touchCount = event.getPointerCount();
        final int x0 = (int) event.getX(0);
        final int y0 = (int) event.getY(0);
        //Log.i(TAG, "Dn:" + touchCount);

        // just got a new single touch
        if (touchCount == 1)
        {
            // get time to detect double taps
            long touchNowMS = System.currentTimeMillis();
            long touchDeltaMS = touchNowMS - lastTouchMS;
            lastTouchMS = touchNowMS;

            if (touchDeltaMS < 250)
                myView.queueEvent(new Runnable() {
                    public void run() {
                        GLES3Lib.onDoubleClick(1, x0, y0);
                    }
                });
            else
                myView.queueEvent(new Runnable() {
                    public void run() {
                        GLES3Lib.onMouseDown(1, x0, y0);
                    }
                });
        }

        // it's two fingers but one delayed (already executed mouse down
        else if (touchCount == 2 && pointersDown == 1) {
            final int x1 = (int) event.getX(1);
            final int y1 = (int) event.getY(1);
            myView.queueEvent(new Runnable() {
                public void run() {
                    GLES3Lib.onMouseUp(1, x0, y0);
                }
            });
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

            if (touchDeltaMS < 250)
                myView.queueEvent(new Runnable() {
                    public void run() {
                        GLES3Lib.onMenuButton();
                    }
                });
            else
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

    public boolean handleTouchUp(final MotionEvent event)
    {
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

    public boolean handleTouchMove(final MotionEvent event)
    {
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
}
