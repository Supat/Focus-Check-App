import Foundation
import AVFoundation
import CoreImage
import UIKit

/// Live back-camera frame source. Mirrors `VideoFrameSource`'s
/// shape externally — `currentImage` republishes per captured frame
/// so the existing renderer pipeline picks each one up — but is
/// driven by an `AVCaptureSession` instead of an `AVPlayer`. No
/// timeline / transport: the only controls are start and stop.
///
/// Pipeline:
///   AVCaptureDevice (back wide-angle camera)
///     → AVCaptureDeviceInput
///       → AVCaptureSession
///         → AVCaptureVideoDataOutput (BGRA, IOSurface-backed)
///           → sample-buffer delegate per captured frame
///             → CIImage → main-actor publish
///
/// Permission: requests camera access at first `start()`. If the
/// user has previously denied, `start()` throws — caller surfaces
/// the error. The app's `.camera(purposeString:)` capability in
/// Package.swift is what lets the OS show the prompt.
///
/// Color: BGRA8 + sRGB tag matches `VideoFrameSource`. iOS hardware
/// captures Rec.709 by default in this format; SDR throughout.
///
/// **Threading:** the class is *not* @MainActor — the AVCapture
/// session lives on `sessionQueue` and orientation tracking flips
/// rotation on the same queue, so making the class main-isolated
/// would force a hop on every property access. Instead, every
/// `@Published` setter explicitly dispatches to main via
/// `DispatchQueue.main.async`, which is what SwiftUI observation
/// requires. `@unchecked Sendable` reflects that the
/// session-owned state is internally serialized via the dispatch
/// queue.
final class CameraFrameSource: NSObject, ObservableObject, @unchecked Sendable {
    /// Most recently captured frame, ready for the renderer. Nil
    /// before the first frame and after `stop()`.
    @Published private(set) var currentImage: CIImage?
    /// Capture-time presentation timestamp. Mostly here for parity
    /// with `VideoFrameSource` so the smoother / analyzer keying
    /// machinery can stay source-agnostic. Wall-clock relative.
    @Published private(set) var currentTime: CMTime = .zero
    /// True after `start()` succeeds, until `stop()` runs.
    @Published private(set) var isRunning: Bool = false

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(
        label: "FocusApp.CameraFrameSource.session"
    )
    private let sampleQueue = DispatchQueue(
        label: "FocusApp.CameraFrameSource.samples"
    )
    private var didConfigure = false
    /// Live device-orientation observer. Subscribed in `start()` so
    /// rotating the iPad mid-session updates the capture connection's
    /// rotation angle and the published frame stays upright.
    private var orientationObserver: NSObjectProtocol?

    override init() {
        super.init()
    }

    /// Request authorization (if needed) and start the capture
    /// session. Idempotent — calling on an already-running source
    /// is a no-op. Throws when permission is denied or no back
    /// camera is available.
    func start() async throws {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            break
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            if !granted { throw CameraFrameSourceError.permissionDenied }
        case .denied, .restricted:
            throw CameraFrameSourceError.permissionDenied
        @unknown default:
            throw CameraFrameSourceError.permissionDenied
        }

        // Configure + start on the dedicated session queue so the
        // setup work doesn't block main. The continuation flips
        // `isRunning` back on main when the session is up.
        try await withCheckedThrowingContinuation {
            (cont: CheckedContinuation<Void, Error>) in
            sessionQueue.async {
                do {
                    try self.configureIfNeeded()
                    if !self.session.isRunning {
                        self.session.startRunning()
                    }
                    DispatchQueue.main.async {
                        self.isRunning = true
                        cont.resume()
                    }
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }

        // Begin tracking device orientation. iOS only emits
        // `orientationDidChangeNotification` while
        // `beginGeneratingDeviceOrientationNotifications` is active,
        // so we explicitly start it here and balance with end in
        // `stop()`. (The OS refcounts these calls.)
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        orientationObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.orientationDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.applyDeviceOrientation()
        }
        applyDeviceOrientation()
    }

    /// Stop the capture session. Drops the last published frame so
    /// the renderer doesn't continue showing a stale image after
    /// the source is torn down.
    func stop() {
        if let observer = orientationObserver {
            NotificationCenter.default.removeObserver(observer)
            orientationObserver = nil
        }
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
            DispatchQueue.main.async {
                self.isRunning = false
                self.currentImage = nil
            }
        }
    }

    /// Push the current `UIDevice.orientation` onto the capture
    /// connection's `videoRotationAngle`. Called on start, on every
    /// `orientationDidChangeNotification`, and from
    /// `configureIfNeeded` to seed the initial value.
    /// Face-up / face-down / unknown orientations are ignored —
    /// the previously-set angle stays in effect (so resting the
    /// iPad flat doesn't rotate the picture).
    private func applyDeviceOrientation() {
        let orientation = UIDevice.current.orientation
        guard let angle = Self.rotationAngle(for: orientation) else { return }
        sessionQueue.async {
            guard let connection = self.videoOutput.connection(with: .video)
            else { return }
            if connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
            }
        }
    }

    /// Map a UIDeviceOrientation to AVCaptureConnection's
    /// `videoRotationAngle` convention (the rotation applied to the
    /// captured frame so it appears upright in that device pose).
    /// Returns nil for face-up / face-down / unknown — caller skips
    /// the update on those.
    private static func rotationAngle(
        for orientation: UIDeviceOrientation
    ) -> CGFloat? {
        switch orientation {
        case .portrait:           return 90
        case .portraitUpsideDown: return 270
        case .landscapeLeft:      return 0
        case .landscapeRight:     return 180
        default:                  return nil
        }
    }

    /// One-time AVCaptureSession setup — pick the back wide-angle
    /// camera, wire it into the session, attach a BGRA video data
    /// output. Reruns are no-ops via the `didConfigure` guard.
    /// Runs on `sessionQueue` (called from `start`).
    private func configureIfNeeded() throws {
        guard !didConfigure else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // .high preset is a sensible default — full 1080p on most
        // iPads, falls back to whatever the device supports. We
        // don't need 4K for a 2 Hz analysis pulse.
        if session.canSetSessionPreset(.high) {
            session.sessionPreset = .high
        }

        // Prefer the back wide-angle (the default 1× lens). On
        // devices without a back camera (e.g. Mac Catalyst), fall
        // through to whatever the system picks for video.
        let device = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .back
        ) ?? AVCaptureDevice.default(for: .video)
        guard let device else {
            throw CameraFrameSourceError.noCamera
        }

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else {
            throw CameraFrameSourceError.cannotAddInput
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String:
                kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String:
                [String: Any](),
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: sampleQueue)
        guard session.canAddOutput(videoOutput) else {
            throw CameraFrameSourceError.cannotAddOutput
        }
        session.addOutput(videoOutput)

        // Seed the capture connection with whatever the current
        // device orientation maps to (portrait fallback when the
        // device is flat / unknown so the first frame isn't
        // sideways). Live updates flow through
        // `applyDeviceOrientation` on rotation events.
        if let connection = videoOutput.connection(with: .video) {
            let initialAngle =
                Self.rotationAngle(for: UIDevice.current.orientation) ?? 90
            if connection.isVideoRotationAngleSupported(initialAngle) {
                connection.videoRotationAngle = initialAngle
            }
        }

        didConfigure = true
    }
}

extension CameraFrameSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Per-frame callback off `sampleQueue`. Wraps the pixel buffer
    /// as a `CIImage` and hops to main to update `@Published`.
    /// `nonisolated` because the protocol's signature is unisolated
    /// — at the publish step we Task-hop back into the actor.
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let image = CIImage(
            cvPixelBuffer: pixelBuffer,
            options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!]
        )
        Task { @MainActor [weak self] in
            self?.currentImage = image
            self?.currentTime = time
        }
    }
}

enum CameraFrameSourceError: Error, LocalizedError {
    case permissionDenied
    case noCamera
    case cannotAddInput
    case cannotAddOutput

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Camera access is required. Enable it in Settings → Privacy → Camera."
        case .noCamera:
            return "No back camera is available on this device."
        case .cannotAddInput, .cannotAddOutput:
            return "Couldn't configure the camera capture session."
        }
    }
}
