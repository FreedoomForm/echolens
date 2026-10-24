package com.echolens.app.vision

import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TrackSnapshot
import com.echolens.app.util.Angles
import kotlin.math.abs
import kotlin.math.tan

/**
 * Converts per-frame detections into world-anchored tracks.
 *
 * Each detection is un-projected into the camera frame (azimuth/elevation
 * from the bounding-box position, distance from the object's typical size),
 * then rotated by the current head yaw into a session-fixed world frame.
 * Objects keep their world position when the head turns — that is what
 * makes the sounds stay in place and the scene feel "external".
 */
class SceneTracker(
    private val memoryMs: Long = 6000L
) {
    // FOV of a typical phone main camera at 4:3, measured in the UPRIGHT portrait frame.
    private val hFovPortrait = 0.846f   // ~48.5° horizontal
    private val vFovPortrait = 1.082f   // ~62°  vertical
    private val tanHalfV = tan(vFovPortrait / 2f)

    private val matchAzRad = 0.35f      // ~20°
    private val matchElevRad = 0.5f     // ~29°
    private val maxTracks = 24

    private class Track(
        val id: Long,
        val label: String,
        val ruName: String,
        val spec: MaterialSpec,
        var azWorld: Float,
        var elevWorld: Float,
        var distance: Float,
        var lastSeenMs: Long,
        var confidence: Float
    )

    private val tracks = ArrayList<Track>(16)
    private var nextId = 1L

    @Synchronized
    fun onDetections(
        detections: List<DetectorEngine.Detection>,
        headYaw: Float,
        headPitch: Float,
        nowMs: Long
    ) {
        val used = HashSet<Long>()
        for (d in detections) {
            val azCam = (d.cxNorm - 0.5f) * hFovPortrait
            val elevCam = (0.5f - d.cyNorm) * vFovPortrait
            val azWorld = Angles.normalize(headYaw + azCam)
            val elevWorld = Angles.clamp(headPitch + elevCam, -1.25f, 1.25f)
            // Monocular distance guess from the object's typical height.
            val distGuess = (d.spec.realHeightM / (2f * tanHalfV * d.hFrac)).coerceIn(0.25f, 12f)

            var best: Track? = null
            for (t in tracks) {
                if (t.label != d.label || used.contains(t.id)) continue
                val dAz = abs(Angles.normalize(azWorld - t.azWorld))
                val dEl = abs(elevWorld - t.elevWorld)
                if (dAz < matchAzRad && dEl < matchElevRad) {
                    best = t
                    break
                }
            }
            if (best != null) {
                // Exponential smoothing keeps the sound anchored and stable.
                best.azWorld = Angles.normalize(best.azWorld + 0.35f * Angles.normalize(azWorld - best.azWorld))
                best.elevWorld += 0.35f * (elevWorld - best.elevWorld)
                best.distance += 0.25f * (distGuess - best.distance)
                best.lastSeenMs = nowMs
                best.confidence = d.score
                used.add(best.id)
            } else {
                val t = Track(nextId++, d.label, d.ruName, d.spec, azWorld, elevWorld, distGuess, nowMs, d.score)
                tracks.add(t)
                used.add(t.id)
                if (tracks.size > maxTracks) {
                    tracks.sortBy { it.lastSeenMs }
                    tracks.removeAt(0)
                }
            }
        }
    }

    /** Returns live tracks and expires stale ones (not seen for [memoryMs]). */
    @Synchronized
    fun snapshot(nowMs: Long): List<TrackSnapshot> {
        tracks.removeAll { nowMs - it.lastSeenMs > memoryMs }
        return tracks.map { t ->
            TrackSnapshot(t.id, t.label, t.ruName, t.spec, t.azWorld, t.elevWorld, t.distance, t.lastSeenMs, t.confidence)
        }
    }

    @Synchronized
    fun clear() {
        tracks.clear()
    }

    @Synchronized
    fun trackCount(): Int = tracks.size
}
