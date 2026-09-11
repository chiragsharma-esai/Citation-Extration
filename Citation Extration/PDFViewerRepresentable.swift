import SwiftUI
import PDFKit

struct PDFViewerRepresentable: NSViewRepresentable {
    let pdfDocument: PDFDocument?
    let highlightText: String?
    let highlightCitation: String?
    let highlightPage: Int?

    func makeNSView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displaysPageBreaks = true
        return pdfView
    }

    func updateNSView(_ pdfView: PDFView, context: Context) {
        if pdfView.document != pdfDocument {
            pdfView.document = pdfDocument
        }

        guard let document = pdfDocument else { return }

        // 1. Target page par scroll karein
        var targetPage: PDFPage? = nil
        if let pageNum = highlightPage, pageNum >= 1, pageNum <= document.pageCount {
            targetPage = document.page(at: pageNum - 1)
            if let targetPage {
                pdfView.go(to: targetPage)
            }
        }

        // 2. Selection Highlight logic (Multi-level fallback)
        var matchedSelection: PDFSelection? = nil

        // Step A: Citation se dhundhein
        if let citation = highlightCitation?.trimmingCharacters(in: .whitespacesAndNewlines), !citation.isEmpty {
            // A1. Exact string search
            let matches = document.findString(citation, withOptions: [.caseInsensitive])
            matchedSelection = filterByPage(matches, targetPage: targetPage)

            // A2. Line Break Fallback: Agar text line break ke paar ho, toh main tokens (e.g. "251 ELT 348") se dhundhein
            if matchedSelection == nil {
                let tokens = citation
                    .components(separatedBy: CharacterSet.alphanumerics.inverted)
                    .filter { $0.count >= 2 }

                if tokens.count >= 2 {
                    let coreQuery = tokens.suffix(3).joined(separator: " ")
                    let coreMatches = document.findString(coreQuery, withOptions: [.caseInsensitive])
                    matchedSelection = filterByPage(coreMatches, targetPage: targetPage)
                }
            }

            // A3. Typo Fallback: Agar "SCC" vs "SSC" ho
            if matchedSelection == nil && citation.contains("SCC") {
                let altCitation = citation.replacingOccurrences(of: "SCC", with: "SSC")
                let altMatches = document.findString(altCitation, withOptions: [.caseInsensitive])
                matchedSelection = filterByPage(altMatches, targetPage: targetPage)
            }

            // 🔹 A4. NEW — Regex-based flexible match directly on page.string.
            //     Handles line-wraps, brackets/punctuation gaps, and reporter-code typos
            //     (e.g. "SSC" vs "SCC") in one shot — same tolerant logic used during extraction.
            if matchedSelection == nil {
                matchedSelection = regexHighlight(
                    in: document,
                    citation: citation,
                    targetPage: targetPage
                )
            }
        }

        // Step B: Agar Citation se highlight na mile, toh Case Name se dhundhein
        if matchedSelection == nil, let caseName = highlightText?.trimmingCharacters(in: .whitespacesAndNewlines), !caseName.isEmpty {
            let nameMatches = document.findString(caseName, withOptions: [.caseInsensitive])
            matchedSelection = filterByPage(nameMatches, targetPage: targetPage)

            if matchedSelection == nil {
                let words = caseName.components(separatedBy: .whitespaces).filter { $0.count > 2 }
                if words.count >= 2 {
                    let shortName = words.prefix(2).joined(separator: " ")
                    let shortMatches = document.findString(shortName, withOptions: [.caseInsensitive])
                    matchedSelection = filterByPage(shortMatches, targetPage: targetPage)
                }
            }
        }

        // 3. Document Viewer me Blue Highlight set karein
        if let selection = matchedSelection {
            pdfView.setCurrentSelection(selection, animate: true)
            pdfView.scrollSelectionToVisible(nil)
        }
    }

    /// Match ko target page ke sath verify karein taaki galat page ka text highlight na ho
    private func filterByPage(_ selections: [PDFSelection], targetPage: PDFPage?) -> PDFSelection? {
        guard let targetPage else { return selections.first }
        for sel in selections {
            if sel.pages.contains(targetPage) {
                return sel
            }
        }
        return selections.first
    }

    /// 🔹 NEW: Flexible regex search directly against a page's own extracted text.
    /// Numeric tokens (year/volume/page-no) match exactly; reporter-code letters (SCC/ELT/ITR etc.)
    /// are wildcarded to tolerate OCR/typo mismatches (e.g. "SSC" vs "SCC"); gaps between tokens
    /// (spaces, line-breaks, brackets, punctuation) are fully flexible — this is what makes
    /// line-wrapped citations like "...Ltd. 2010\n(251) ELT 348 (Mad.)" still match.
    private func regexHighlight(in document: PDFDocument, citation: String, targetPage: PDFPage?) -> PDFSelection? {
        let tokens = citation
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
        guard tokens.count >= 2 else { return nil }

        let gap = "[\\s\\(\\)\\[\\]\\.,\\-–/\\n]+"
        let fuzzyTokens = tokens.map { token -> String in
            let isNumeric = token.allSatisfy { $0.isNumber }
            let isShortAlpha = !isNumeric && token.count <= 6 && token.allSatisfy { $0.isLetter }
            if isShortAlpha {
                return "[A-Za-z]{2,6}"
            }
            return NSRegularExpression.escapedPattern(for: token)
        }
        let pattern = fuzzyTokens.joined(separator: gap) + "[\\)\\]\\.]{0,3}"

        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return nil
        }

        // Search order: target page first, then whole document as a fallback
        var pagesToSearch: [PDFPage] = []
        if let targetPage { pagesToSearch.append(targetPage) }
        for i in 0..<document.pageCount {
            if let page = document.page(at: i), page !== targetPage {
                pagesToSearch.append(page)
            }
        }

        for page in pagesToSearch {
            guard let pageText = page.string as NSString? else { continue }
            let fullRange = NSRange(location: 0, length: pageText.length)
            if let match = regex.firstMatch(in: pageText as String, range: fullRange) {
                if let selection = page.selection(for: match.range) {
                    return selection
                }
            }
        }
        return nil
    }
}
