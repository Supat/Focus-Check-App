import Foundation
import ImageIO
import CoreGraphics

/// Compact read-only EXIF snapshot for display. Fields are nil when the source
/// file didn't include them (screenshots, heavily edited images, some RAW
/// formats). Populated via `read(from:)` off a URL — cheap, no pixel decode.
struct ExposureInfo: Equatable {
    var focalLengthMM: Float?
    var focalLengthEquivalent35MM: Float?
    var exposureTimeSeconds: Float?
    var fNumber: Float?
    var iso: Int?
    /// EXIF SubjectDistance, in metres. Rarely populated by phone cameras;
    /// DSLRs with modern AF modules write it more reliably.
    var subjectDistanceMeters: Float?

    static func read(from url: URL) -> ExposureInfo? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        else { return nil }
        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any] ?? [:]

        var info = ExposureInfo()
        info.focalLengthMM = (exif[kCGImagePropertyExifFocalLength] as? NSNumber)?.floatValue
        info.focalLengthEquivalent35MM = (exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber)?.floatValue
        info.exposureTimeSeconds = (exif[kCGImagePropertyExifExposureTime] as? NSNumber)?.floatValue
        info.fNumber = (exif[kCGImagePropertyExifFNumber] as? NSNumber)?.floatValue
        if let iso = exif[kCGImagePropertyExifISOSpeedRatings] as? [NSNumber],
           let first = iso.first {
            info.iso = first.intValue
        }
        info.subjectDistanceMeters = (exif[kCGImagePropertyExifSubjectDistance] as? NSNumber)?.floatValue
        return info.isEmpty ? nil : info
    }

    /// "1/250 s" for fast shutters, "0.5 s" for long ones.
    var formattedShutter: String? {
        guard let s = exposureTimeSeconds, s > 0 else { return nil }
        if s >= 1 { return String(format: "%.1f s", s) }
        let denom = (1.0 / s).rounded()
        return "1/\(Int(denom)) s"
    }

    /// Prefer the 35mm-equivalent focal length when available; else the native
    /// sensor focal length with one decimal below 10 mm.
    var formattedFocalLength: String? {
        if let eq = focalLengthEquivalent35MM, eq > 0 {
            return "\(Int(eq.rounded())) mm"
        }
        if let fl = focalLengthMM, fl > 0 {
            return fl >= 10 ? "\(Int(fl.rounded())) mm" : String(format: "%.1f mm", fl)
        }
        return nil
    }

    /// "1.2 m" for close subjects, "12 m" past 10 m, "∞" for far/infinity
    /// markers (some cameras encode infinity as a very large number).
    var formattedFocusDistance: String? {
        guard let d = subjectDistanceMeters, d > 0 else { return nil }
        if d >= 1000 { return "∞" }
        if d >= 10 { return "\(Int(d.rounded())) m" }
        return String(format: "%.1f m", d)
    }

    /// True when no EXIF fields at all were populated.
    private var isEmpty: Bool {
        focalLengthMM == nil &&
        focalLengthEquivalent35MM == nil &&
        exposureTimeSeconds == nil &&
        fNumber == nil &&
        iso == nil &&
        subjectDistanceMeters == nil
    }
}
