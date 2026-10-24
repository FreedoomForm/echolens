package com.echolens.app.head

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import com.echolens.app.util.Angles
import kotlin.math.asin
import kotlin.math.atan2

/**
 * Tracks where the back camera points, in a world frame that stays fixed
 * for the whole session (relative yaw — no magnetic reference, immune to
 * indoor magnetic interference).
 *
 * Mount assumption: the phone is strapped to the head, screen facing the
 * user, back camera looking forward, top of the phone pointing up.
 *
 * yaw:   radians; 0 = direction at session start; positive when the head
 *        turns right.
 * pitch: radians; positive when the camera looks up.
 */
class HeadTracker(context: Context) : SensorEventListener {

    @Volatile var yaw = 0f
        private set

    @Volatile var pitch = 0f
        private set

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var sensor: Sensor? = null
    private var handlerThread: HandlerThread? = null
    private val rotationMatrix = FloatArray(9)
    private var baseYaw: Float? = null

    fun start() {
        if (handlerThread != null) return
        // GAME_ROTATION_VECTOR: gyro-integrated, no magnetic distortion indoors.
        sensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        val s = sensor ?: return
        val thread = HandlerThread("HeadTracker").also { it.start() }
        handlerThread = thread
        sensorManager.registerListener(this, s, SensorManager.SENSOR_DELAY_GAME, Handler(thread.looper))
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        handlerThread?.quitSafely()
        handlerThread = null
        sensor = null
        baseYaw = null
    }

    override fun onSensorChanged(event: SensorEvent) {
        val s = sensor ?: return
        if (event.sensor.type != s.type) return
        if (event.values.size < 4) return
        SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
        // v_world = R * v_device; the camera optical axis is -Z in device coords,
        // so the camera forward direction in world coords is minus the 3rd column.
        val fx = -rotationMatrix[2]
        val fy = -rotationMatrix[5]
        val fz = -rotationMatrix[8]
        var y = atan2(fx, fy)
        val p = asin(Angles.clamp(fz, -1f, 1f))
        if (baseYaw == null) baseYaw = y
        y -= baseYaw ?: 0f
        yaw = Angles.normalize(y)
        pitch = p
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
