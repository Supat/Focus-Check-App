import SwiftUI
import MetalKit

#if canImport(AppKit)
typealias PlatformViewRepresentable = NSViewRepresentable
#else
typealias PlatformViewRepresentable = UIViewRepresentable
#endif

/// Bridges an `MTKView` into SwiftUI for both iPadOS and macOS.
///
/// The renderer is held strongly on the coordinator so its lifetime tracks the view.
/// Drawable-size changes during Stage Manager resize only trigger a renderer update
/// when the delta exceeds 1 pixel (CLAUDE.md rule).
struct MetalView: PlatformViewRepresentable {
    @ObservedObject var viewModel: FocusViewModel

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

    #if canImport(AppKit)
    func makeNSView(context: Context) -> MTKView { makeMTKView(context: context) }
    func updateNSView(_ nsView: MTKView, context: Context) { updateMTKView(nsView) }
    #else
    func makeUIView(context: Context) -> MTKView { makeMTKView(context: context) }
    func updateUIView(_ uiView: MTKView, context: Context) { updateMTKView(uiView) }
    #endif

    private func makeMTKView(context: Context) -> MTKView {
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
        }
        view.delegate = context.coordinator
        return view
    }

    private func updateMTKView(_ view: MTKView) {
        // Renderer observes the view model directly. Loop runs at 60 fps via `isPaused = false`.
    }
}
