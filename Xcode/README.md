# FocusApp — Xcode project

An Xcode-buildable version of the same source tree that powers
`FocusApp.swiftpm`. Both projects compile the files under
`../FocusApp.swiftpm/Sources/App/` — pick whichever environment you
prefer and any code change flows to both.

| | `.swiftpm` (Swift Playgrounds) | `.xcodeproj` (this folder) |
|---|---|---|
| Runs in | Swift Playgrounds on iPad / Mac | Xcode on Mac |
| Signing | Unsigned / development | Signed with your Developer account |
| Distribution | Sideload via Playgrounds | TestFlight / App Store |
| `SCSensitivityAnalyzer` | Stays `.disabled` | Works once user grants access |
| App Store capabilities | Limited | Full |

## One-time setup (requires a Mac with Homebrew)

```bash
cd Xcode
./setup.sh
```

The script installs [XcodeGen](https://github.com/yonaskolb/XcodeGen)
via `brew` if needed, then generates `FocusApp.xcodeproj` from
`project.yml`.

Re-run `./setup.sh` any time you add / remove files or change
`project.yml`. It's idempotent.

## After generating

1. Open `FocusApp.xcodeproj` in Xcode.
2. Select the *FocusApp* target → *Signing & Capabilities* → pick
   your Team. This sets the bundle ID to `com.<team>.FocusApp`.
   (The default in `project.yml` is `com.example.FocusApp`, which
   won't sign.)
3. Drop a 1024×1024 PNG into
   `Resources/Assets.xcassets/AppIcon.appiconset/` (rename to match
   the `Contents.json` filename field) for a real app icon. Placeholder
   is empty otherwise.
4. **Optional but needed for the sensitive-content mosaic:** under
   *Signing & Capabilities* add the **Sensitive Content Analysis**
   capability. Without this, `analysisPolicy` stays `.disabled` even
   in signed builds.
5. Build & Run. Target an iPad simulator / device running iPadOS 18.1+.

## What's synthesised into the Info.plist

XcodeGen's `info.properties` block in `project.yml` inlines the
required keys so the generated project doesn't need a separate
`Info.plist` file:

- `NSPhotoLibraryUsageDescription` — system prompt text when the
  app first reaches into Photos for original filenames.
- `NSSensitiveContentAnalysisUsageDescription` — prompt text when
  the app first calls `SCSensitivityAnalyzer` on a signed build.
  macOS only shows this via its per-app Sensitive Content Warning
  toggle; iOS ties it to Communication Safety in Screen Time.
- `UIDeviceFamily = [2]` — iPad only. The Xcode app runs on Apple
  Silicon Macs automatically via *Designed for iPad*.
- `UIRequiredDeviceCapabilities = [metal]` — match the `.swiftpm`'s
  Metal requirement.
- `UILaunchScreen` — empty dict (blank launch screen).

## Dependencies

`ZIPFoundation` is declared as a Swift Package dependency in
`project.yml` under `packages:`. Xcode resolves it on first build.
The depth model and NSFW model are runtime downloads (they don't
live in the app bundle) — nothing further to configure here.

## Going back to the Playground

`FocusApp.swiftpm/` is untouched by anything in this folder. Open
it in Swift Playgrounds and continue iterating there any time; the
source changes still flow to this Xcode project the next time you
build.
