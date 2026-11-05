import SwiftUI

/// Consistent, quiet chrome for image tools and compact option buttons.
struct StudioButtonStyle: ButtonStyle {
    var emphasized = false
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .medium))
            .lineLimit(1)
            .padding(.horizontal, 10)
            .frame(minHeight: 30)
            .foregroundStyle(emphasized ? Color.accentColor : Color.primary)
            .background {
                RoundedRectangle(cornerRadius: 8)
                    .fill(emphasized ? Color.accentColor.opacity(configuration.isPressed ? 0.24 : 0.13)
                          : Color.primary.opacity(configuration.isPressed ? 0.12 : 0.05))
            }
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(emphasized ? Color.accentColor.opacity(0.3) : Color.primary.opacity(0.1))
            }
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .opacity(isEnabled ? 1 : 0.4)
    }
}

/// One-line note under a control: an icon, a severity tint and, optionally, a longer explanation behind "Why?".
struct StudioNote: View {
    enum Kind { case info, warning, error, unsupported }
    var kind: Kind
    var text: String
    var detail: String? = nil
    var detailTitle = "Why?"
    @State private var showDetail = false

    init(_ kind: Kind, _ text: String, detail: String? = nil, detailTitle: String = "Why?") {
        self.kind = kind
        self.text = text
        self.detail = detail
        self.detailTitle = detailTitle
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: symbol).font(.caption).foregroundStyle(tint)
            // No vertical fixedSize: outside a ScrollView (the generate bar) it makes the split view measure this
            // text at a near-zero width, and the resulting height pushed the whole window's content off screen.
            Text(text)
                .lineLimit(6)
                .textSelection(.enabled)
            if let detail {
                Button(detailTitle) { showDetail.toggle() }
                    .buttonStyle(.link).font(.caption)
                    .popover(isPresented: $showDetail, arrowEdge: .bottom) {
                        Text(detail).font(.callout).padding(14).frame(width: 340)
                            .fixedSize(horizontal: false, vertical: true)
                    }
            }
        }
        .font(.caption)
        .foregroundStyle(kind == .info || kind == .unsupported ? .secondary : tint)
        .padding(.horizontal, kind == .unsupported ? 0 : 8)
        .padding(.vertical, kind == .unsupported ? 0 : 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            if kind != .unsupported {
                RoundedRectangle(cornerRadius: 6).fill(tint.opacity(kind == .info ? 0.06 : 0.12))
            }
        }
    }

    private var symbol: String {
        switch kind {
        case .info: return "info.circle"
        case .warning: return "exclamationmark.triangle"
        case .error: return "xmark.octagon"
        case .unsupported: return "nosign"
        }
    }

    private var tint: Color {
        switch kind {
        case .info: return .secondary
        case .warning: return .orange
        case .error: return .red
        case .unsupported: return .secondary
        }
    }
}
