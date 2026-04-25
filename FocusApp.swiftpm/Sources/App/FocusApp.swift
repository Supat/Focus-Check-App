import SwiftUI

/// Cross-view notifications for keyboard-shortcut commands. Posted
/// by the Scene-level `.commands` block (which is what iPadOS 26's
/// new menu bar surfaces shortcut hints from) and consumed by
/// `ImageImporter`'s `.onReceive` listeners. Routing through
/// NotificationCenter avoids threading a shared @ObservableObject
/// between the App scene and a deeply-nested toolbar view just to
/// flip two booleans.
extension Notification.Name {
    static let openPhotoLibrary = Notification.Name("FocusApp.OpenPhotoLibrary")
    static let openFileImporter = Notification.Name("FocusApp.OpenFileImporter")
}

@main
struct FocusApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .windowResizability(.contentSize)
        #endif
        // Scene-level commands surface in iPadOS 26's top menu bar
        // and on Mac via Designed for iPad. `replacing: .newItem`
        // overrides the built-in File menu's empty "New" group so
        // our two Open commands land where users expect File →
        // Open. Hidden Buttons inside ImageImporter previously
        // carried these shortcuts, but iPadOS 26 doesn't surface
        // those in its menu bar; this Scene-level binding is the
        // modern home.
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Photo Library") {
                    NotificationCenter.default.post(name: .openPhotoLibrary, object: nil)
                }
                .keyboardShortcut("o", modifiers: .command)

                Button("Open File…") {
                    NotificationCenter.default.post(name: .openFileImporter, object: nil)
                }
                .keyboardShortcut("o", modifiers: [.command, .shift])
            }
        }
    }
}
