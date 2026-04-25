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
            // `.resizeAspect` preserves the bitmap's proportions
            // when the layer bounds animate during a paused full-
            // screen transition. The default `.resize` would
            // stretch non-uniformly (entering FS makes the image
            // look vertically elongated; exiting compresses it).
            // With aspect-fit, newly-exposed area shows as black
            // briefly until the unpaused renderer redraws at the
            // final drawable size — much less visually jarring
            // than a distorted image.
            layer.contentsGravity = .resizeAspect
        }
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        // Renderer observes the view model directly. Loop runs at 60 fps via `isPaused = false`.
        guard uiView.isPaused != isPaused else { return }
        if isPaused {
            // Going into the transition: freeze the drawable so
            // Core Animation can scale the existing bitmap with
            // .resizeAspect while the parent's layout animation
            // grows / shrinks the layer bounds.
            uiView.autoResizeDrawable = false
            uiView.isPaused = true
        } else {
            // Coming out: the layer has already animated to its
            // final bounds, but autoResizeDrawable was off the
            // whole time so MTKView never noticed. Re-enabling
            // alone won't sync — it only kicks in on the next
            // bounds change. Force the drawable to the current
            // bounds so the Coordinator's drawableSizeWillChange
            // fires and the next 60 fps draw fills the new size
            // correctly. Without this the .resizeAspect letterbox
            // from the transition stays painted on screen.
            uiView.autoResizeDrawable = true
            let scale = uiView.layer.contentsScale
            let target = CGSize(
                width: uiView.bounds.width * scale,
                height: uiView.bounds.height * scale
            )
            if target.width > 0 && target.height > 0,
               target != uiView.drawableSize {
                uiView.drawableSize = target
            }
            uiView.isPaused = false
        }
    }
}
