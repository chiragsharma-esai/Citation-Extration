import Foundation
import AppKit
import PDFKit
import CoreText
import UniformTypeIdentifiers

class ExportManager {

    /// Safely escapes a CSV field (if it contains a comma, quote, or newline)
    private static func csvEscape(_ field: String) -> String {
        if field.contains(",") || field.contains("\"") || field.contains("\n") {
            return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return field
    }

    static func saveAsCSV(citedCases: [CitedCase]?, rawContent: String) {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.commaSeparatedText]
        savePanel.nameFieldStringValue = "Extracted_Citations.csv"

        guard savePanel.runModal() == .OK, let url = savePanel.url else { return }

        var csvText: String

        if let citedCases, !citedCases.isEmpty {
            // Structured data available — creating real columns
            var rows = ["Case Name,Citation,Year,Court,Context"]
            for c in citedCases {
                let row = [
                    csvEscape(c.caseName),
                    csvEscape(c.citation ?? ""),
                    c.year.map(String.init) ?? "",
                    csvEscape(c.court ?? ""),
                    csvEscape(c.context ?? "")
                ].joined(separator: ",")
                rows.append(row)
            }
            csvText = rows.joined(separator: "\n")
        } else {
            // Fallback — structured parsing failed, putting raw text into a single column
            csvText = "Extracted Output\n" + csvEscape(rawContent)
        }

        do {
            try csvText.write(to: url, atomically: true, encoding: .utf8)
            NSWorkspace.shared.open(url)  // Opens in Excel (or the default app) immediately after saving
        } catch {
            print("Error saving CSV: \(error)")
        }
    }

    static func saveAsPDF(citedCases: [CitedCase]?, rawContent: String) {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.pdf]
        savePanel.nameFieldStringValue = "Extracted_Citations.pdf"

        guard savePanel.runModal() == .OK, let url = savePanel.url else { return }

        // If structured data is available, convert it into readable formatted text,
        // otherwise just write the raw content
        let content: String
        if let citedCases, !citedCases.isEmpty {
            content = citedCases.enumerated().map { index, c in
                var block = "\(index + 1). \(c.caseName)"
                if let citation = c.citation { block += "\nCitation: \(citation)" }
                if let year = c.year { block += "\nYear: \(year)" }
                if let court = c.court { block += "\nCourt: \(court)" }
                if let context = c.context { block += "\nContext: \(context)" }
                return block
            }.joined(separator: "\n\n")
        } else {
            content = rawContent
        }

        // A4-ish page size in points
        let pageWidth: CGFloat = 595.0
        let pageHeight: CGFloat = 842.0
        let margin: CGFloat = 40.0
        var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)

        let pdfData = NSMutableData()
        guard let consumer = CGDataConsumer(data: pdfData as CFMutableData),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            print("Error: Could not create PDF context")
            return
        }

        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: 12),
            .foregroundColor: NSColor.black
        ]
        let attributedText = NSAttributedString(string: content, attributes: attributes)
        let framesetter = CTFramesetterCreateWithAttributedString(attributedText)

        let textRect = CGRect(x: margin, y: margin,
                               width: pageWidth - margin * 2,
                               height: pageHeight - margin * 2)
        let path = CGMutablePath()
        path.addRect(textRect)

        var currentRange = CFRange(location: 0, length: 0)
        let totalLength = attributedText.length

        repeat {
            context.beginPDFPage(nil)

            let frame = CTFramesetterCreateFrame(framesetter, currentRange, path, nil)
            CTFrameDraw(frame, context)

            let visibleRange = CTFrameGetVisibleStringRange(frame)
            currentRange = CFRange(location: currentRange.location + visibleRange.length, length: 0)

            context.endPDFPage()
        } while currentRange.location < totalLength

        context.closePDF()

        do {
            try pdfData.write(to: url, options: .atomic)
            NSWorkspace.shared.open(url)  // Opens in the PDF viewer immediately after saving
        } catch {
            print("Error saving PDF: \(error)")
        }
    }
}
