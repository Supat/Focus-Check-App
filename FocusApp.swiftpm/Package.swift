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
                .photoLibrary(purposeString: "Select photos to analyze focus regions."),
                .fileAccess(.userSelectedFiles, mode: .readOnly)
            ]
        )
    ],
    targets: [
        .executableTarget(
            name: "App",
            path: "Sources/App",
            resources: [
                // Depth Anything v2 is an optional hybrid-mode asset. Enable by:
                //   1. xcrun coremlcompiler compile DepthAnythingV2SmallF16.mlpackage /tmp/
                //   2. cp -r /tmp/DepthAnythingV2SmallF16.mlmodelc Sources/App/Resources/
                //   3. Uncomment the line below.
                // `.copy` is REQUIRED — `.process` silently breaks `.mlmodelc` directories.
                // .copy("Resources/DepthAnythingV2SmallF16.mlmodelc")
            ]
        )
    ]
)
