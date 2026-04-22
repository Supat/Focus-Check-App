#!/usr/bin/env bash
# Generate FocusApp.xcodeproj from the XcodeGen manifest.
# Run on a Mac with Homebrew available. Creates / refreshes the Xcode
# project in place. Safe to re-run; it reads project.yml each time.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v xcodegen >/dev/null 2>&1; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "error: neither xcodegen nor brew found." >&2
        echo "Install Homebrew (https://brew.sh) or install XcodeGen manually:" >&2
        echo "  https://github.com/yonaskolb/XcodeGen#installing" >&2
        exit 1
    fi
    echo "Installing XcodeGen via Homebrew..."
    brew install xcodegen
fi

xcodegen generate

cat <<'EOF'

 ✓ Generated FocusApp.xcodeproj
   Open it in Xcode, set your Team under Signing & Capabilities,
   then Run. The app builds from ../FocusApp.swiftpm/Sources/App/
   so edits to the Playgrounds sources show up here automatically.
EOF
