import SwiftUI
import PDFKit

struct PDFViewerPanel: View {
    let pdfDocument: PDFDocument?
    let highlightText: String?
    let highlightCitation: String?
    let highlightPage: Int?

    var body: some View {
        VStack(spacing: 0) {
            // Header bar
            HStack(spacing: 10) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Text("Document Viewer")
                    .font(.headline)
                    .foregroundColor(.primary)

                Spacer()

                if let doc = pdfDocument {
                    Text("\(doc.pageCount) page\(doc.pageCount == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color.secondary.opacity(0.12))
                        )
                }

                if let page = highlightPage {
                    Text("p. \(page)")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.accentColor)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            Capsule()
                                .fill(Color.accentColor.opacity(0.12))
                        )
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(.regularMaterial)

            Divider()

            // PDF content
            if pdfDocument != nil {
                PDFViewerRepresentable(
                    pdfDocument: pdfDocument,
                    highlightText: highlightText,
                    highlightCitation: highlightCitation,
                    highlightPage: highlightPage
                )
            } else {
                VStack(spacing: 12) {
                    Spacer()
                    Image(systemName: "doc.text")
                        .font(.system(size: 36))
                        .foregroundColor(.secondary.opacity(0.5))
                    Text("No document loaded")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.08), radius: 4, x: 0, y: 2)
        .transition(.move(edge: .trailing).combined(with: .opacity))
    }
}
