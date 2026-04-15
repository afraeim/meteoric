/*!
 * Permissions Module
 *
 * Exposes Tauri commands for querying and requesting macOS privacy permissions
 * required by Meteroic (Accessibility and Screen Recording), plus the pure-logic
 * helper that decides whether the onboarding screen must be shown.
 *
 * Architecture: thin command wrappers (excluded from coverage) delegate to
 * small, testable functions. The only logic exercised at test-time is
 * `needs_onboarding`, which is a pure predicate with no OS side-effects.
 */

// ─── Pure Logic ──────────────────────────────────────────────────────────────

/// Returns `true` when at least one required permission has not been granted.
///
/// Both Accessibility (hotkey listener) and Screen Recording (/screen command)
/// must be granted for Meteroic to function fully. If either is missing the
/// onboarding screen is shown instead of the normal overlay.
pub fn needs_onboarding(accessibility: bool, screen_recording: bool) -> bool {
    !accessibility || !screen_recording
}

// ─── macOS Permission Checks ─────────────────────────────────────────────────

#[cfg(target_os = "macos")]
#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

/// Returns whether the process currently has Accessibility permission.
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn is_accessibility_granted() -> bool {
    unsafe { AXIsProcessTrusted() }
}

/// Returns whether the process currently has Screen Recording permission.
///
/// Uses `CGPreflightScreenCaptureAccess`, which only returns `true` after
/// a full restart post-grant (unlike `CGWindowListCopyWindowInfo` which
/// returns non-null immediately but before pixels are actually accessible).
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn is_screen_recording_granted() -> bool {
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightScreenCaptureAccess() -> bool;
    }
    unsafe { CGPreflightScreenCaptureAccess() }
}

// ─── Tauri Commands ──────────────────────────────────────────────────────────

/// Returns whether Accessibility permission has been granted.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_accessibility_permission() -> bool {
    is_accessibility_granted()
}

/// Non-macOS stub: Accessibility permission is not managed via this command.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_accessibility_permission() -> bool {
    true
}

/// Opens System Settings to the Accessibility privacy pane so the user can
/// enable the permission without encountering the native system popup.
///
/// This gives a consistent onboarding experience: both Accessibility and
/// Screen Recording are granted via System Settings rather than native dialogs.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn open_accessibility_settings() -> Result<(), String> {
    std::process::Command::new("open")
        .arg(
            "x-apple.systempreferences:com.apple.preference.security\
             ?Privacy_Accessibility",
        )
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Non-macOS stub: no Accessibility settings deep-link is available.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn open_accessibility_settings() -> Result<(), String> {
    Err("Accessibility settings are only supported on macOS".to_string())
}

/// Returns whether Screen Recording permission has been granted.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_screen_recording_permission() -> bool {
    is_screen_recording_granted()
}

/// Non-macOS stub: screen recording permission is platform-specific.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_screen_recording_permission() -> bool {
    true
}

/// Opens System Settings to the Screen Recording privacy pane so the user
/// can enable the permission without navigating there manually.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn open_screen_recording_settings() -> Result<(), String> {
    std::process::Command::new("open")
        .arg(
            "x-apple.systempreferences:com.apple.preference.security\
             ?Privacy_ScreenCapture",
        )
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Non-macOS stub: no Screen Recording settings deep-link is available.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn open_screen_recording_settings() -> Result<(), String> {
    Err("Screen Recording settings are only supported on macOS".to_string())
}

/// Registers Meteroic in the Screen Recording privacy pane and shows the macOS
/// permission prompt.
///
/// `CGRequestScreenCaptureAccess` is the only API that both adds the app to
/// System Settings > Privacy & Security > Screen & System Audio Recording and
/// triggers the native "allow screen recording" alert. Without calling this
/// first, Meteroic will not appear in the Screen Recording list at all.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn request_screen_recording_access() {
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGRequestScreenCaptureAccess() -> bool;
    }
    unsafe {
        CGRequestScreenCaptureAccess();
    }
}

/// Non-macOS stub: permission prompt is macOS-only.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn request_screen_recording_access() {}

/// Returns `true` if Screen & System Audio Recording permission is currently
/// granted. Delegates to `CGPreflightScreenCaptureAccess`, which correctly
/// returns `false` when the permission has not been granted, fixing the
/// historical false-positive from `CGWindowListCopyWindowInfo(0, 0)`.
///
/// Called by PermissionsStep during onboarding polling so the "Quit & Reopen"
/// prompt appears once the user toggles the permission on in System Settings.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_screen_recording_tcc_granted() -> bool {
    is_screen_recording_granted()
}

/// Non-macOS stub: TCC does not exist outside macOS.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn check_screen_recording_tcc_granted() -> bool {
    true
}

/// Quits Meteroic and immediately relaunches it.
///
/// Called after the user grants Screen Recording permission. macOS requires
/// a full process restart before the new permission takes effect.
///
/// Writes "intro" to the DB before restarting so `notify_frontend_ready`
/// shows the intro screen on the next launch without calling any permission
/// API. Permission APIs (CGPreflightScreenCaptureAccess) can return stale
/// results immediately after a restart on macOS 15+; trusting the DB stage
/// avoids that unreliability entirely.
#[tauri::command]
#[cfg(target_os = "macos")]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn quit_and_relaunch(app_handle: tauri::AppHandle, db: tauri::State<crate::history::Database>) {
    if let Ok(conn) = db.0.lock() {
        let _ = crate::onboarding::set_stage(&conn, &crate::onboarding::OnboardingStage::Intro);
    }
    app_handle.restart();
}

/// Non-macOS stub: relaunch flow is currently only used by macOS onboarding.
#[tauri::command]
#[cfg(not(target_os = "macos"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub fn quit_and_relaunch(
    app_handle: tauri::AppHandle,
    _db: tauri::State<crate::history::Database>,
) {
    app_handle.restart();
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn needs_onboarding_false_when_both_granted() {
        assert!(!needs_onboarding(true, true));
    }

    #[test]
    fn needs_onboarding_true_when_accessibility_missing() {
        assert!(needs_onboarding(false, true));
    }

    #[test]
    fn needs_onboarding_true_when_screen_recording_missing() {
        assert!(needs_onboarding(true, false));
    }

    #[test]
    fn needs_onboarding_true_when_both_missing() {
        assert!(needs_onboarding(false, false));
    }
}
