import SwiftUI
import MetalKit

/// Bridges an `MTKView` into SwiftUI.
///
/// This app's product type is `.iOSApplication`, so UIKit is always the runtime
/// view system — even when the app runs on macOS via "Designed for iPad".
/// No AppKit branch is needed or correct.
///
/// The renderer is held strongly on the coordinator so its lifetime tracks the
/// view. Drawable-size changes during Stage Manager resize only trigger a
/// renderer update when the delta exceeds 1 pixel (CLAUDE.md rule).
///
/// `isPaused` lets a parent (e.g. ContentView during a full-screen toggle)
/// freeze rendering while a SwiftUI layout animation is in flight. While
/// paused we also disable `autoResizeDrawable` so the existing bitmap is
/// scaled by Core Animation rather than wiped to black on each layer-bounds
/// step — gives a smooth visual stretch with no main-thread contention from
/// per-frame composites.
struct MetalView: UIViewRepresentable {
    @ObservedObject var viewModel: FocusViewModel
    var isPaused: Bool = false

    final class Coordinator: NSObject, MTKViewDelegate {
        let renderer: FocusRenderer
        var lastSize: CGSize = .zero

        init(renderer: FocusRenderer) {
            self.renderer = renderer
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            if abs(size.width - lastSize.width) > 1 || abs(size.height - lastSize.height) > 1 {
                lastSize = size
                renderer.resize(to: size)
            }
        }

        func draw(in view: MTKView) {
            renderer.draw(in: view)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: FocusRenderer(viewModel: viewModel))
    }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.framebufferOnly = false
        view.autoResizeDrawable = true
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.preferredFramesPerSecond = 60
        view.colorPixelFormat = .rgba16Float
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        if let layer = view.layer as? CAMetalLayer {
            layer.wantsExtendedDynamicRangeContent = true
            layer.pixelFormat = .rgba16Float
            // `.resize` lets Core Animation stretch the existing
            // drawable smoothly when we pause + freeze auto-resize
            // for a layout transition. Default value already, but
            // pin it explicitly so a future config change doesn't
            // regress the full-screen smoothness.
            layer.contentsGravity = .resize
        }
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // Renderer observes the view model directly. Loop runs at 60 fps via `isPaused = false`.
        if uiView.isPaused != isPaused {
            uiView.isPaused = isPaused
            // Freeze the drawable while paused so Core Animation
            // can scale the existing bitmap during the parent's
            // layout animation. Re-enabling triggers
            // drawableSizeWillChange + a fresh draw at the new size.
            uiView.autoResizeDrawable = !isPaused
        }
    }
}
