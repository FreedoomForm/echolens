package com.echolens.app.core

/** Sound-producing material category assigned to a detected object class. */
enum class Material {
    WOOD, METAL, GLASS, CERAMIC, ENGINE, WATER, ELECTRONICS,
    FABRIC, PLANT, PAPER, PERSON, ANIMAL, BIRD, FOOD, NEUTRAL
}

/**
 * Everything the app needs to know about one object class:
 * the material it is "made of" (its sound texture), its Russian name,
 * a typical visible height (for monocular distance estimation),
 * navigation importance (0..1) and the reference distance for loudness falloff.
 */
class MaterialSpec(
    val material: Material,
    val ruName: String,
    val realHeightM: Float,
    val importance: Float,
    val refDistanceM: Float
)

/** Head pose in a session-local world frame (yaw zero = direction at start). */
class HeadPose(val yaw: Float, val pitch: Float)

/** A world-anchored object track as consumed by the audio engine. */
class TrackSnapshot(
    val id: Long,
    val label: String,
    val ruName: String,
    val spec: MaterialSpec,
    /** Azimuth in the session world frame, radians. Positive = right of start direction. */
    val azWorld: Float,
    /** Elevation, radians. Positive = above horizon. */
    val elevWorld: Float,
    /** Estimated distance, meters. */
    val distance: Float,
    val lastSeenMs: Long,
    val confidence: Float
)
