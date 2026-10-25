package com.echolens.app.vision

import com.echolens.app.core.MaterialSpec
import com.echolens.app.core.TrackSnapshot
import com.echolens.app.util.Angles
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.tan

/**
 * Converts per-frame detections into world-anchored tracks.
 *
 * Each detection is un-projected into the camera frame (azimuth/elevation
 * from the bounding-box position, distance from the object's typical size),
 * then rotated by the current head yaw into a session-fixed world frame.
 * Objects keep their world position when the head turns — that is what
 * makes the sounds stay in place and the scene feel "external".
 *
 * Distance is fused from two sources:
 *  - bbox-size prior (object class height vs box height)
 *  - Depth Anything V2 relative depth (higher = closer), calibrated
 *    against the bbox prior by a global scale factor.
 */
class SceneTracker(
    private val memoryMs: Long = 6000L
) {
    // FOV of a typical phone main camera at 4:3, measured in the UPRIGHT portrait frame.
    private val hFovPortrait = 0.846f   // ~48.5° horizontal
    private val vFovPortrait = 1.082f   // ~62°  vertical
    private val tanHalfV = tan(vFovPortrait / 2f)

    private val matchAzRad = 0.26f      // ~15°
    private val matchElevRad = 0.42f    // ~24°
    private val maxTracks = 24
    private val labelSwitchVotes = 3

    // Global depth calibration: distance ≈ depthScale / relativeDepth.
    private var depthScale = 0f

    private class Track(
        val id: Long,
        var label: String,
        var ruName: String,
        var spec: MaterialSpec,
        var azWorld: Float,
        var elevWorld: Float,
        var distance: Float,
        var lastSeenMs: Long,
        var confidence: Float
    ) {
        /** Consecutive frames voting for a different label (hysteresis). */
        var pendingLabel: String? = null
        var pendingVotes = 0
    }

    private val tracks = ArrayList<Track>(16)
    private var nextId = 1L

    @Synchronized
    fun onDetections(
        detections: List<YoloDetector.Detection>,
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
            // Prior: monocular distance guess from the object's typical height.
            val bboxDist = (d.spec.realHeightM / (2f * tanHalfV * d.hFrac)).coerceIn(0.25f, 15f)

            // Fuse with relative depth when available.
            val dist: Float = if (d.relDepth != null && d.relDepth > 1e-3f) {
                val product = bboxDist * d.relDepth
                depthScale = if (depthScale <= 0f) product else depthScale + 0.12f * (product - depthScale)
                val depthDist = (depthScale / max(d.relDepth, 1e-3f)).coerceIn(0.25f, 15f)
                // Trust depth more once the scale has settled.
                0.35f * bboxDist + 0.65f * depthDist
            } else {
                bboxDist
            }

            var best: Track? = null
            // Pass 1: same-label match (the common, stable case).
            for (t in tracks) {
                if (used.contains(t.id) || t.label != d.label) continue
                val dAz = abs(Angles.normalize(azWorld - t.azWorld))
                val dEl = abs(elevWorld - t.elevWorld)
                if (dAz < matchAzRad && dEl < matchElevRad) {
                    best = t
                    break
                }
            }
            // Pass 2: any-label match in a tighter window — prevents the same
            // object from spawning a second track when the detector flickers
            // between similar classes (cup/bowl, chair/table, cat/dog).
            if (best == null) {
                for (t in tracks) {
                    if (used.contains(t.id)) continue
                    val dAz = abs(Angles.normalize(azWorld - t.azWorld))
                    val dEl = abs(elevWorld - t.elevWorld)
                    if (dAz < matchAzRad * 0.6f && dEl < matchElevRad * 0.6f) {
                        best = t
                        break
                    }
                }
            }
            if (best != null) {
                // Exponential smoothing keeps the sound anchored and stable.
                best.azWorld = Angles.normalize(best.azWorld + 0.35f * Angles.normalize(azWorld - best.azWorld))
                best.elevWorld += 0.35f * (elevWorld - best.elevWorld)
                best.distance += 0.25f * (dist - best.distance)
                best.lastSeenMs = nowMs
                best.confidence = d.score
                applyLabelHysteresis(best, d)
                used.add(best.id)
            } else {
                val t = Track(nextId++, d.label, d.ruName, d.spec, azWorld, elevWorld, dist, nowMs, d.score)
                tracks.add(t)
                used.add(t.id)
                if (tracks.size > maxTracks) {
                    tracks.sortBy { it.lastSeenMs }
                    tracks.removeAt(0)
                }
            }
        }
    }

    /**
     * Switches a track's label only after several consecutive confident
     * detections of the new class — single-frame flicker no longer changes
     * what the object sounds like or how it is announced.
     */
    private fun applyLabelHysteresis(t: Track, d: YoloDetector.Detection) {
        if (d.label == t.label) {
            t.pendingLabel = null
            t.pendingVotes = 0
            return
        }
        if (d.score < t.confidence - 0.10f) {
            // A weaker detection of a different class is likely noise.
            return
        }
        if (t.pendingLabel == d.label) {
            t.pendingVotes++
        } else {
            t.pendingLabel = d.label
            t.pendingVotes = 1
        }
        if (t.pendingVotes >= labelSwitchVotes) {
            t.label = d.label
            t.ruName = d.ruName
            t.spec = d.spec
            t.pendingLabel = null
            t.pendingVotes = 0
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

}
