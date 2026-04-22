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
                .fileAccess(.userSelectedFiles, mode: .readOnly)
            ]
        )
    ],
    dependencies: [
        // Pure-Swift ZIP extraction — used by DepthModelDownloader to unzip the
        // compiled `.mlmodelc` after downloading it at runtime.
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
