import Foundation
import AppKit
import UniformTypeIdentifiers

class ExportManager {

    private static func csvEscape(_ field: String) -> String {
        if field.contains(",") || field.contains("\"") || field.contains("\n") {
            return "\"" + field.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return field
    }

    static func saveAsCSV(citedCases: [CitedCase]) {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.commaSeparatedText]
        savePanel.nameFieldStringValue = "Extracted_Citations.csv"

        guard savePanel.runModal() == .OK, let url = savePanel.url else { return }

        var rows = ["Case Name,Citation,Year,Court,Context,Page Number"]
        for c in citedCases {
            let row = [
                csvEscape(c.caseName),
                csvEscape(c.citation ?? ""),
                c.year.map(String.init) ?? "",
                csvEscape(c.court ?? ""),
                csvEscape(c.context ?? ""),
                c.pageNumber.map(String.init) ?? ""
            ].joined(separator: ",")
            rows.append(row)
        }

        let csvText = rows.joined(separator: "\n")

        do {
            try csvText.write(to: url, atomically: true, encoding: .utf8)
            NSWorkspace.shared.open(url)
        } catch {
            print("Error saving CSV: \(error)")
        }
    }
}
