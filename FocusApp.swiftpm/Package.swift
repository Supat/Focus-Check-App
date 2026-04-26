// swift-tools-version: 5.9

import PackageDescription
import AppleProductTypes

let package = Package(
    name: "FocusApp",
    defaultLocalization: "en",
    platforms: [
        .iOS("18.1"),
        .macOS("14.0")
    ],
    products: [
        .iOSApplication(
            name: "FocusApp",
            targets: ["App"],
            bundleIdentifier: "com.example.FocusApp",
            teamIdentifier: "",
            displayVersion: "1.0",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .camera),
            accentColor: .presetColor(.orange),
            supportedDeviceFamilies: [.pad, .mac],
            supportedInterfaceOrientations: [
                .portrait,
                .landscapeRight,
                .landscapeLeft,
                .portraitUpsideDown(.when(deviceFamilies: [.pad]))
            ],
            capabilities: [
                .photoLibrary(purposeString: "Select photos to analyze focus regions and display their filename."),
                .fileAccess(.userSelectedFiles, mode: .readOnly),
                // Required for the live-camera source — back-camera
                // frames feed the same renderer + analysis pipeline
                // as imported videos. Without this capability the OS
                // refuses the AVCaptureDevice request.
                .camera(purposeString: "The live camera feed is analyzed for sensitive content and overlaid with redaction in real time."),
                // Required for runtime model downloads (depth, NSFW). Without
                // this the macOS app sandbox denies URLSession requests with
                // 'server with the specified hostname could not be found'.
                .outgoingNetworkConnections()
            ]
        )
    ],
    dependencies: [
        // Pure-Swift ZIP extraction — used by ModelArchiveInstaller to unzip
        // the compiled `.mlmodelc` after downloading it at runtime.
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.19")
    ],
    targets: [
        .executableTarget(
            name: "App",
            dependencies: [
                .product(name: "ZIPFoundation", package: "ZIPFoundation")
            ],
            path: "Sources/App"
        )
    ]
)
