/*
 * Copyright (C) 2016 mendhak
 *
 * This file is part of GPSLogger for Android.
 *
 * GPSLogger for Android is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * GPSLogger for Android is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GPSLogger for Android.  If not, see <http://www.gnu.org/licenses/>.
 */

package ch.fhnw.comgr;

import android.location.Location;
import android.location.LocationListener;
import android.location.LocationProvider;
import android.os.Bundle;
import android.util.Log;


class GeneralLocationListener implements LocationListener {

    private static String _listenerName;
    private static GLES3Activity _activity;
    private static final String TAG = "SLProject";
    protected String latestHdop;
    protected String latestPdop;
    protected String latestVdop;
    protected String geoIdHeight;
    protected String ageOfDgpsData;
    protected String dgpsId;
    protected int satellitesUsedInFix;

    GeneralLocationListener(GLES3Activity activity, String name) {
        _activity = activity;
        _listenerName = name;
    }

    /**
     * Event raised when a new fix is received.
     */
    public void onLocationChanged(Location loc) {
        if (loc != null) {
            _activity.onLocationChanged(loc);
        }
    }

    public void onProviderDisabled(String provider) {
        _activity.locationSensorRestart();
    }

    public void onProviderEnabled(String provider) {
        _activity.locationSensorRestart();
    }

    public void onStatusChanged(String provider, int status, Bundle extras) {
        if (status == LocationProvider.OUT_OF_SERVICE) {
            Log.i(TAG, provider + " is out of service");
            _activity.locationSensorStop();
        }

        if (status == LocationProvider.AVAILABLE) {
            //Log.i(TAG, provider + " is available");
        }

        if (status == LocationProvider.TEMPORARILY_UNAVAILABLE) {
            Log.i(TAG, "onStatusChanged:" + provider + " is temporarily unavailable");
        }
    }
}
