/**
 * Tunable constants for the live procedure workstation.
 *
 * Everything here is a *display* parameter, not a clinical standard. The one
 * externally-grounded value is TOTAL_PROCEDURE_TARGET_S — see its comment.
 */

/**
 * Target total procedure duration, in seconds.
 *
 * ESGE performance measures set a minimum examination time of >= 7 minutes
 * (intubation to extubation) for diagnostic upper GI endoscopy, and longer
 * inspection is consistently associated with higher detection of clinically
 * significant lesions.
 *
 * NOTE: this is a *total* procedure target. There is no established per-region
 * target for esophagus/stomach/duodenum, so we deliberately do not invent one —
 * per-region figures are reported as a share of this total, never against a
 * fabricated regional goal.
 */
export const TOTAL_PROCEDURE_TARGET_S = 7 * 60;

/**
 * Largest gap between consecutive samples that still counts as continuous time.
 *
 * The video is paused by backpressure whenever the backend falls behind, and the
 * user can scrub, so wall clock and video clock diverge. A gap beyond this (or a
 * backwards jump) is treated as a discontinuity: no dwell is credited and the
 * region anchor resets. 4x the capture interval.
 */
export const MAX_SAMPLE_GAP_S = 2.0;

/** EMA weight for the "recent" score used by the visibility readout. */
export const RECENT_SCORE_ALPHA = 0.2;

/** Consecutive frames that must agree before the displayed score changes. */
export const SCORE_SETTLE_FRAMES = 2;

/** Minimum time the displayed score is held before it may change again (ms). */
export const SCORE_SETTLE_HOLD_MS = 1200;

/** Motion smoothing — mirrors the thresholds the previous UI used. */
export const MOTION_CONFIDENCE_THRESHOLD = 0.6;
export const MOTION_CONSENSUS_FRAMES = 3;

/** Below this mean confidence the visibility readout reports nothing at all. */
export const VISIBILITY_CONFIDENCE_GATE = 0.5;

/** No result within this many ms and the workstation stops asserting liveness. */
export const STALE_AFTER_MS = 2500;

/**
 * A sent frame is written off after this long.
 *
 * Without an expiry, one dropped reply leaves the in-flight count permanently at
 * the backpressure threshold, the video stays paused forever, and the UI looks
 * healthy while nothing is happening.
 */
export const IN_FLIGHT_TIMEOUT_MS = 6000;

/** Filmstrip. */
export const FILMSTRIP_MAX_RETAINED = 24;
export const FILMSTRIP_VISIBLE_SLOTS = 3;
export const FILMSTRIP_MIN_GAP_S = 5;
export const THUMB_WIDTH = 240;
export const THUMB_QUALITY = 0.5;
/** Frames kept for look-back, since a result arrives after its frame is gone. */
export const THUMB_RING_SIZE = 8;

/**
 * ESGE station tracking.
 *
 * Availability is latched: one usable landmark frame turns the checklist on, but
 * it only turns off after this many CONSECUTIVE unsupported/error frames. Per-frame
 * layout checks fail on the odd frame (scope outside the patient, red-out, menus),
 * and a checklist that vanishes on one bad frame is worse than none. Skipped
 * frames (landmark_every_n >= 2) are not evaluated and neither count nor reset.
 */
export const STATION_AVAILABILITY_RELEASE_FRAMES = 6;
/**
 * Full-resolution analysis frames kept for the station gallery, keyed by ticket.
 * Only has to cover the frames in flight when a result arrives.
 */
export const STATION_GALLERY_RING = 8;
/** Quiet period before the "N of 10 stations observed" announcement is spoken. */
export const STATION_ANNOUNCE_DEBOUNCE_MS = 1500;

/** Coaching prompt timing. Asymmetric arm/clear windows create hysteresis. */
export const COACH_TIMING = {
  ARM_FRAMES: 3,
  MIN_DWELL_MS: 2500,
  CLEAR_MS: 2000,
  RATE_LIMIT_MS: 3000,
  REPEAT_MS: 20000,
} as const;

export const ALERT_TIMING = {
  ARM_FRAMES: 2,
  MIN_DWELL_MS: 3000,
  CLEAR_MS: 1500,
  RATE_LIMIT_MS: 0,
  REPEAT_MS: 0,
} as const;
