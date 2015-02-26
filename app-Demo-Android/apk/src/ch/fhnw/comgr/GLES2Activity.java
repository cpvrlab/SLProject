/*
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

package ch.fhnw.comgr;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.util.DisplayMetrics;
import android.view.Display;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.view.View;
import android.view.View.OnTouchListener;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class GLES2Activity extends Activity implements OnTouchListener, SensorEventListener
{
    GLES2View       myView;				// OpenGL view
    static  int     pointersDown = 0;	// NO. of fingers down
	static  long    lastTouchMS = 0;	// Time of last touch in ms
	private SensorManager 	mSensorManager;

    @Override protected void onCreate(Bundle icicle) 
	{
		Log.i("SLProject", "GLES2Activity.onCreate");
        super.onCreate(icicle);
		        
		// Extract (unzip) files in APK
		try 
		{	Log.i("SLProject", "extractAPK");
			GLES2Lib.App = getApplication();
			GLES2Lib.extractAPK();
		} catch (IOException e) 
		{
			Log.e("SLProject", "Error extracting files from the APK archive: " + e.getMessage());
		}
		
		// Create view
		myView = new GLES2View(GLES2Lib.App);
		GLES2Lib.view = myView;
		myView.setOnTouchListener(this);
		Log.i("SLProject", "setContentView");
		setContentView(myView);
		
		// Get display resolution. This is used to scale the menu buttons accordingly
		DisplayMetrics metrics = new DisplayMetrics();
		getWindowManager().getDefaultDisplay().getMetrics(metrics);
        int dpi = (int)(((float)metrics.xdpi + (float)metrics.ydpi) * 0.5);
		GLES2Lib.dpi = dpi;
		Log.i("SLProject", "DisplayMetrics: " + dpi);
		
		// Init Sensor
		mSensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
	}

    @Override protected void onPause() 
	{
		Log.i("SLProject", "GLES2Activity.onPause");
		super.onPause();	
		myView.onPause();	
		myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onClose();}});
		finish();
		
        mSensorManager.unregisterListener(this);
		
		Log.i("SLProject", "System.exit(0)");
		System.exit(0);
		//android.os.Process.killProcess(android.os.Process.myPid());
    }

    @Override protected void onResume() 
	{
		Log.i("SLProject", "GLES2Activity.onResume");
		super.onResume();
		mSensorManager.registerListener(this,
										mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
										SensorManager.SENSOR_DELAY_FASTEST);
		myView.onResume();
	}

    @Override protected void onStop() 
	{
		Log.i("SLProject", "GLES2Activity.onStop");
		super.onStop();
		System.exit(0);
	}

    @Override protected void onDestroy() 
	{
		Log.i("SLProject", "GLES2Activity.onDestroy");
		super.onDestroy();
    }

	/**
	* Events:
	* 
	* Finger Down
	* -----------
	* Just tap on screen -> onMouseDown, onMouseUp
	* Tap and hold       -> onMouseDown
	* 			release   -> onMouseUp
	* 2 Fingers same time -> onTouch2Down
	* 2 Fingers not same time -> onMouseDown, onMouseUp, onTouch2Down
	* 
	* Finger Up
	* ---------
	* 2 Down, release one -> onTouch2Up
	*         release other one -> onMouseUp
	* 2 Down, release one, put another one down -> onTouch2Up, onTouch2Down
	* 2 Down, release both same time -> onTouch2Up
	* 2 Down, release both not same time -> onTouch2Up
	*/

	public boolean handleTouchDown(final MotionEvent event) 
	{
		int touchCount = event.getPointerCount();
        final int x0 = (int)event.getX(0);
        final int y0 = (int)event.getY(0);
		//Log.i("SLProject", "Dn:" + touchCount);
		
		// just got a new single touch
		if (touchCount == 1) {
			// get time to detect double taps
			long touchNowMS = System.currentTimeMillis();
			long touchDeltaMS = touchNowMS - lastTouchMS;
			lastTouchMS = touchNowMS;
			
			if (touchDeltaMS < 250)
				myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onDoubleClick(1, x0, y0);}});
			else
				myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onMouseDown(1, x0, y0);}});
		}
		
		// it's two fingers but one delayed (already executed mouse down
		else if (touchCount == 2 && pointersDown == 1) {
            final int x1 = (int)event.getX(1);
            final int y1 = (int)event.getY(1);
 			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onMouseUp(1, x0, y0);}});
 			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onTouch2Down(x0, y0 ,x1, y1);}});
		// it's two fingers at the same time
		} else if (touchCount == 2) {
            // get time to detect double taps
			long touchNowMS = System.currentTimeMillis();
			long touchDeltaMS = touchNowMS - lastTouchMS;
			lastTouchMS = touchNowMS;

            final int x1 = (int)event.getX(1);
            final int y1 = (int)event.getY(1);
            
			if (touchDeltaMS < 250)
 			    myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onMenuButton();}});
            else
                myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onTouch2Down(x0, y0 ,x1, y1);}});
		}
		pointersDown = touchCount;
		myView.requestRender();
		return true;
	}

	public boolean handleTouchUp(final MotionEvent event) 
	{
		int touchCount = event.getPointerCount();
		//Log.i("SLProject", "Up:" + touchCount + " x: " + (int)event.getX(0) + " y: " + (int)event.getY(0));
        final int x0 = (int)event.getX(0);
        final int y0 = (int)event.getY(0);
		if (touchCount == 1) {
			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onMouseUp(1, x0, y0);}});
        }
		else if (touchCount == 2) {
            final int x1 = (int)event.getX(1);
            final int y1 = (int)event.getY(1);
			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onTouch2Up(x0, y0 ,x1, y1);}});
        }
		
		pointersDown = touchCount;
		myView.requestRender();
		return true;
	}

	public boolean handleTouchMove(final MotionEvent event) 
	{
		final int x0 = (int)event.getX(0);
		final int y0 = (int)event.getY(0);
	 	int touchCount = event.getPointerCount();
		//Log.i("SLProject", "Mv:" + touchCount);
		
	 	if (touchCount == 1) {
			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onMouseMove(x0, y0);}});
	 	}
        else if (touchCount == 2) {
            final int x1 = (int)event.getX(1);
            final int y1 = (int)event.getY(1);
            myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onTouch2Move(x0, y0, x1, y1);}});
		}
		myView.requestRender();
		return true;
    }

	@Override
	public boolean onTouch(View v, final MotionEvent event) 
	{
		if (event == null)
        {   Log.i("SLProject", "onTouch: null event");
            return false;
        }
        
        int action = event.getAction();
		int actionCode = action & MotionEvent.ACTION_MASK;
		
		try
		{	if (actionCode == MotionEvent.ACTION_DOWN || 
                actionCode == MotionEvent.ACTION_POINTER_DOWN)
				return handleTouchDown(event);
			else if (actionCode == MotionEvent.ACTION_UP || 
                     actionCode == MotionEvent.ACTION_POINTER_UP)
				return handleTouchUp(event);
			else if (actionCode == MotionEvent.ACTION_MOVE)
				return handleTouchMove(event);
			else Log.i("SLProject", "Unhandeled Event: " + actionCode);
		}
		catch(Exception ex)
		{	Log.i("SLProject",  "onTouch (Exception: " + actionCode);
		}
		
		return false;
	}
    
    @Override
	public boolean onCreateOptionsMenu(Menu menu) 
	{
		Log.i("SLProject", "Menu Button pressed");
		myView.queueEvent(new Runnable() 
		{
			public void run() {GLES2Lib.onMenuButton();}
		});
		return false;
	}

	public void onAccuracyChanged(Sensor sensor, int accuracy) 
	{	
		Log.i("SLProject", String.format("onAccuracyChanged"));
    }

    public void onSensorChanged(SensorEvent event)
	{
		if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR)
		{
			// The ROTATION_VECTOR sensor is a virtual fusion sensor
			// The quality strongly depends on the underlying algorithm and on
			// the sensor manufacturer. (See also chapter 7 in the book:
			// "Professional Sensor Programming (WROX Publishing)"
			
			// Get 3x3 rotation matrix from XYZ-rotation vector (see docs)
			float R[] = new float[9];
			SensorManager.getRotationMatrixFromVector(R , event.values);
			
			// Get yaw, pitch & roll rotation angles in radians from rotation matrix
			float[] YPR = new float[3]; 
			SensorManager.getOrientation(R, YPR);

			// Check display orientation (a preset orientation is set in the AndroidManifext.xml)
			Display display = getWindowManager().getDefaultDisplay();
			if(display.getWidth() < display.getHeight())
			{	// Map pitch, yaw and roll to portrait display orientation
				final float p = YPR[1] * -1.0f - (float)Math.PI*0.5f;
				final float y = YPR[0] * -1.0f;
				final float r = YPR[2] * -1.0f;				
				myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onRotationPYR(p,y,r);}});
			} else 
			{	// Map pitch, yaw and roll to landscape display orientation for Oculus Rift conformance
				final float p = YPR[2] * -1.0f - (float)Math.PI*0.5f;
				final float y = YPR[0] * -1.0f;
				final float r = YPR[1];
				myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onRotationPYR(p,y,r);}});
			}

			/*
			// Get the rotation quaternion from the XYZ-rotation vector (see docs)
			final float Q[] = new float[4];
			SensorManager.getQuaternionFromVector(Q, event.values);
			myView.queueEvent(new Runnable() {public void run() {GLES2Lib.onRotationQUAT(Q[1],Q[2],Q[3],Q[0]);}});
			*/
		}
    }
}
