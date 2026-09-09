import SwiftUI
import PDFKit

struct PDFViewerRepresentable: NSViewRepresentable {
    let pdfDocument: PDFDocument?
    let highlightText: String?
    let highlightCitation: String?
    let highlightPage: Int?

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> PDFView {
        let pdfView = PDFView()
        pdfView.autoScales = true
        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.backgroundColor = NSColor(white: 0.12, alpha: 1.0)
        pdfView.document = pdfDocument
        return pdfView
    }

    func updateNSView(_ pdfView: PDFView, context: Context) {
        if pdfView.document !== pdfDocument {
            pdfView.document = pdfDocument
            context.coordinator.removeAllHighlights()
            context.coordinator.lastHighlightID = nil
        }

        let currentID = highlightID
        guard currentID != context.coordinator.lastHighlightID else { return }
        context.coordinator.lastHighlightID = currentID

        context.coordinator.removeAllHighlights()

        guard let document = pdfView.document else { return }

        if let pageNum = highlightPage, pageNum >= 1, pageNum <= document.pageCount {
            let pageIndex = pageNum - 1

            if let selection = findBestSelection(in: document, pageIndex: pageIndex) {
                context.coordinator.addHighlight(for: selection)
                scrollTo(selection: selection, in: pdfView)
            } else if let page = document.page(at: pageIndex) {
                let destination = PDFDestination(page: page, at: NSPoint(x: 0, y: page.bounds(for: .mediaBox).maxY))
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    pdfView.go(to: destination)
                }
            }
        } else if let selection = findBestSelection(in: document, pageIndex: nil) {
            context.coordinator.addHighlight(for: selection)
            scrollTo(selection: selection, in: pdfView)
        }
    }

    private var highlightID: String? {
        guard highlightText != nil || highlightCitation != nil else { return nil }
        return "\(highlightText ?? "")||\(highlightCitation ?? "")||\(highlightPage ?? -1)"
    }

    private func scrollTo(selection: PDFSelection, in pdfView: PDFView) {
        guard let firstPage = selection.pages.first else { return }
        let bounds = selection.bounds(for: firstPage)
        let destination = PDFDestination(page: firstPage, at: NSPoint(x: 0, y: bounds.maxY + 60))
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            pdfView.go(to: destination)
        }
    }

    /// Finds the citation, searching ONLY the pages it could legitimately be on.
    ///
    /// This used to fall back to `allSelections.first` — the first match anywhere in
    /// the document. With a case cited on several pages that meant clicking the
    /// occurrence on page 6 scrolled to page 34, because page 34 happened to hold
    /// the first match. A row must never scroll somewhere the case was not cited.
    ///
    /// The allowed pages are the target page plus the one after it, since a case
    /// name can straddle a page break. If nothing matches there we return nil, and
    /// the caller scrolls to the correct page without a highlight — being on the
    /// right page unhighlighted beats being on the wrong page highlighted.
    private func findBestSelection(in document: PDFDocument, pageIndex: Int?) -> PDFSelection? {
        let candidates = buildSearchCandidates()

        guard let pageIndex else {
            // No page information at all — only then may we search the whole document.
            for candidate in candidates {
                let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                if let first = document.findString(trimmed, withOptions: .caseInsensitive).first {
                    return first
                }
            }
            return nil
        }

        let allowedPages: [PDFPage] = [pageIndex, pageIndex + 1]
            .filter { $0 >= 0 && $0 < document.pageCount }
            .compactMap { document.page(at: $0) }
        guard !allowedPages.isEmpty else { return nil }

        // Try every candidate string on the target page before allowing the
        // straddle page, so the exact page always wins.
        for page in allowedPages {
            for candidate in candidates {
                let trimmed = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                if let sel = findOnPage(page, text: trimmed),
                   let s = sel.string, !s.isEmpty {
                    return sel
                }
            }
        }

        return nil
    }

    private func findOnPage(_ page: PDFPage, text: String) -> PDFSelection? {
        guard let pageContent = page.string, !pageContent.isEmpty else { return nil }
        if pageContent.localizedCaseInsensitiveContains(text) {
            let sels = page.document?.findString(text, withOptions: .caseInsensitive) ?? []
            return sels.first { $0.pages.contains(page) }
        }
        return nil
    }

    private func buildSearchCandidates() -> [String] {
        var candidates: [String] = []

        if let text = highlightText {
            candidates.append(text)

            let vSeparators = [" v. ", " v ", " vs. ", " vs ", " V. ", " V "]
            for sep in vSeparators {
                if let range = text.range(of: sep, options: .caseInsensitive) {
                    let firstParty = String(text[text.startIndex..<range.lowerBound])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let secondParty = String(text[range.upperBound...])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    if !firstParty.isEmpty { candidates.append(firstParty) }
                    if !secondParty.isEmpty { candidates.append(secondParty) }
                    break
                }
            }
        }

        if let citation = highlightCitation, !citation.isEmpty {
            candidates.append(citation)
        }

        return candidates
    }

    class Coordinator {
        var lastHighlightID: String?
        private var activeAnnotations: [(page: PDFPage, annotation: PDFAnnotation)] = []

        func addHighlight(for selection: PDFSelection) {
            for page in selection.pages {
                let bounds = selection.bounds(for: page)
                let annotation = PDFAnnotation(bounds: bounds, forType: .highlight, withProperties: nil)
                annotation.color = NSColor.controlAccentColor.withAlphaComponent(0.35)
                page.addAnnotation(annotation)
                activeAnnotations.append((page: page, annotation: annotation))
            }
        }

        func removeAllHighlights() {
            for entry in activeAnnotations {
                entry.page.removeAnnotation(entry.annotation)
            }
            activeAnnotations.removeAll()
        }
    }
}
