import SwiftUI

/// The playground palette, 1:1 with renderer/styles.css `:root`.
enum Theme {
    static let bg = Color(hex: 0x16181D)
    static let panel = Color(hex: 0x20242C)
    static let panel2 = Color(hex: 0x272C36)
    static let edge = Color(hex: 0x3A4150)
    static let text = Color(hex: 0xD7DCE5)
    static let muted = Color(hex: 0x8A93A3)
    static let jack = Color(hex: 0x4A5365)
    static let jackHot = Color(hex: 0xCDD6E6)
    static let wire = Color(hex: 0x6F7B92)
    static let err = Color(hex: 0xF08A8A)
    static let dotGrid = Color(hex: 0x232733)

    static let transportOff = Color(hex: 0x2C6E4A)
    static let transportOn = Color(hex: 0x8A3C3C)
    static let clockOn = Color(hex: 0x2C5A6E)
    static let scrubAccent = Color(hex: 0x4FD6C4)   // reversing — the moat gesture

    static let mono = Font.system(size: 13, design: .monospaced)
    static let monoSmall = Font.system(size: 11, design: .monospaced)
    static let monoTiny = Font.system(size: 10, design: .monospaced)
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255
        )
    }
}
