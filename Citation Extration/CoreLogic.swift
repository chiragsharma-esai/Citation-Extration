import Foundation
import SwiftUI
import Combine
import PDFKit
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

// MARK: - Generation Result Container
struct GenerationResult {
    let responseText: String
    let thinkingText: String
}

// MARK: - Data Models
struct ChatHistory: Identifiable, Hashable {
    let id = UUID()
    var title: String
    var pdfText: String
    var pdfURL: URL?
    var extractedOutput: String
    var generationTimeText: String
    var thinkingText: String = ""
    var thinkingEnabled: Bool = true
    var model: LLMModel
}

// MARK: - Structured Extraction Schema
// MARK: - Structured Extraction Schema
struct CitedCase: Codable, Identifiable, Hashable {
    let id = UUID()
    let caseName: String
    let citation: String?
    let year: Int?
    let court: String?
    let context: String?
    let pageNumber: Int?
    let isSelfReference: Bool
    let pdfHighlightText: String?   // raw text span used for PDF search/highlight

    private enum CodingKeys: String, CodingKey {
        case caseName, citation, year, court, context, pageNumber, isSelfReference, pdfHighlightText
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        caseName = (try? container.decode(String.self, forKey: .caseName)) ?? "Unknown Case"
        citation = Self.decodeLenientString(container, .citation)
        court = Self.decodeLenientString(container, .court)
        context = Self.decodeLenientString(container, .context)
        year = Self.decodeLenientInt(container, .year)
        pageNumber = Self.decodeLenientInt(container, .pageNumber)
        isSelfReference = (try? container.decode(Bool.self, forKey: .isSelfReference)) ?? false
        pdfHighlightText = Self.decodeLenientString(container, .pdfHighlightText)
    }

    init(caseName: String, citation: String?, year: Int?, court: String?, context: String?, pageNumber: Int? = nil, isSelfReference: Bool = false, pdfHighlightText: String? = nil) {
        self.caseName = caseName
        self.citation = citation
        self.year = year
        self.court = court
        self.context = context
        self.pageNumber = pageNumber
        self.isSelfReference = isSelfReference
        self.pdfHighlightText = pdfHighlightText ?? citation
    }

    private static func decodeLenientString(_ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys) -> String? {
        if let value = try? container.decode(String.self, forKey: key) {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty || trimmed.lowercased() == "null" || trimmed.lowercased() == "n/a" {
                return nil
            }
            return trimmed
        }
        return nil
    }

    private static func decodeLenientInt(_ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys) -> Int? {
        if let value = try? container.decode(Int.self, forKey: key) {
            return value
        }
        if let stringValue = try? container.decode(String.self, forKey: key) {
            let trimmed = stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return Int(trimmed)
        }
        return nil
    }
}

struct DocumentExtractionResult: Codable {
    let citedCases: [CitedCase]

    static func parse(from rawOutput: String) -> DocumentExtractionResult? {
        var text = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)

        if let fenceStart = text.range(of: "```json") {
            text = String(text[fenceStart.upperBound...])
        } else if let fenceStart = text.range(of: "```") {
            text = String(text[fenceStart.upperBound...])
        }
        if let fenceEnd = text.range(of: "```") {
            text = String(text[..<fenceEnd.lowerBound])
        }

        if let candidate = extractDataObject(from: text) {
            if let data = candidate.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(DocumentExtractionResult.self, from: data) {
                return decoded
            }

            let quoted = quoteBareKeys(candidate)
            if let data = quoted.data(using: .utf8),
               let decoded = try? JSONDecoder().decode(DocumentExtractionResult.self, from: data) {
                return decoded
            }
        }

        if let arrayText = extractBareArray(from: text) {
            for variant in [arrayText, quoteBareKeys(arrayText)] {
                if let data = variant.data(using: .utf8),
                   let cases = try? JSONDecoder().decode([CitedCase].self, from: data) {
                    return DocumentExtractionResult(citedCases: cases)
                }
            }
        }

        return nil
    }

    private static func extractBareArray(from text: String) -> String? {
        let chars = Array(text)
        guard let start = chars.firstIndex(of: "[") else { return nil }

        var depth = 0
        var inString = false
        var escaped = false

        for k in start..<chars.count {
            let c = chars[k]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
                continue
            }
            if c == "\"" { inString = true; continue }
            if c == "[" { depth += 1 }
            if c == "]" {
                depth -= 1
                if depth == 0 { return String(chars[start...k]) }
            }
        }
        return nil
    }

    private static func extractDataObject(from text: String) -> String? {
        let chars = Array(text)

        var keyIndex: Int? = nil
        let needle = Array("citedCases")
        var i = 0
        while i + needle.count <= chars.count {
            if Array(chars[i..<(i + needle.count)]) == needle {
                var j = i + needle.count
                while j < chars.count, chars[j] == "\"" || chars[j].isWhitespace { j += 1 }
                if j < chars.count, chars[j] == ":" {
                    j += 1
                    while j < chars.count, chars[j].isWhitespace { j += 1 }
                    if j < chars.count, chars[j] == "[" {
                        keyIndex = i
                        break
                    }
                }
            }
            i += 1
        }
        guard let foundKey = keyIndex else { return nil }

        var start = foundKey
        while start >= 0, chars[start] != "{" { start -= 1 }
        guard start >= 0 else { return nil }

        var stack: [Character] = []
        var inString = false
        var escaped = false
        var end: Int? = nil

        var k = start
        while k < chars.count {
            let c = chars[k]
            if inString {
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
            } else {
                switch c {
                case "\"": inString = true
                case "{": stack.append("}")
                case "[": stack.append("]")
                case "}", "]":
                    if stack.last == c { stack.removeLast() }
                    if stack.isEmpty { end = k }
                default: break
                }
            }
            if end != nil { break }
            k += 1
        }

        if let end {
            return String(chars[start...end])
        }

        var repaired = String(chars[start...])
        if inString { repaired += "\"" }
        while let closer = stack.popLast() { repaired.append(closer) }
        return repaired
    }

    private static func quoteBareKeys(_ json: String) -> String {
        var result = ""
        var inString = false
        var escaped = false
        let chars = Array(json)
        var i = 0

        while i < chars.count {
            let c = chars[i]

            if inString {
                result.append(c)
                if escaped { escaped = false }
                else if c == "\\" { escaped = true }
                else if c == "\"" { inString = false }
                i += 1
                continue
            }

            if c == "\"" {
                inString = true
                result.append(c)
                i += 1
                continue
            }

            if c == "{" || c == "," {
                result.append(c)
                i += 1

                var whitespace = ""
                while i < chars.count, chars[i].isWhitespace {
                    whitespace.append(chars[i])
                    i += 1
                }

                var identifier = ""
                while i < chars.count, chars[i].isLetter || chars[i].isNumber || chars[i] == "_" {
                    identifier.append(chars[i])
                    i += 1
                }

                var lookahead = i
                while lookahead < chars.count, chars[lookahead].isWhitespace { lookahead += 1 }

                if !identifier.isEmpty, lookahead < chars.count, chars[lookahead] == ":" {
                    result += whitespace + "\"" + identifier + "\""
                } else {
                    result += whitespace + identifier
                }
                continue
            }

            result.append(c)
            i += 1
        }

        return result
    }
}

struct DocumentSelfReference {
    let caseTitle: String?
    let caseNumber: String?

    static func parse(from rawOutput: String) -> DocumentSelfReference {
        var text = rawOutput.trimmingCharacters(in: .whitespacesAndNewlines)

        if let fenceStart = text.range(of: "```json") {
            text = String(text[fenceStart.upperBound...])
        } else if let fenceStart = text.range(of: "```") {
            text = String(text[fenceStart.upperBound...])
        }
        if let fenceEnd = text.range(of: "```") {
            text = String(text[..<fenceEnd.lowerBound])
        }

        guard let firstBrace = text.firstIndex(of: "{"),
              let lastBrace = text.lastIndex(of: "}"),
              let data = String(text[firstBrace...lastBrace]).data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return DocumentSelfReference(caseTitle: nil, caseNumber: nil)
        }

        let title = (json["caseTitle"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let number = (json["caseNumber"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return DocumentSelfReference(
            caseTitle: (title?.isEmpty ?? true) ? nil : title,
            caseNumber: (number?.isEmpty ?? true) ? nil : number
        )
    }
}

enum ModelArchitectureType {
    case llm
    case vlm
}

enum LLMModel: String, CaseIterable, Identifiable {
    case gemma4 = "Gemma 4 E2B"
    case gemma3n = "Gemma 3n E2B"
    case qwen35 = "Qwen 3.5 2B"
    case qwen25 = "Qwen 2.5 1.5B"
    case addModel = "Add New Model..."

    var id: String { rawValue }

    var hubRepoID: String? {
        switch self {
        case .gemma4: return "mlx-community/gemma-4-e2b-it-4bit"
        case .gemma3n: return "mlx-community/gemma-3n-E2B-it-lm-4bit"
        case .qwen35: return "mlx-community/Qwen3.5-2B-4bit"
        case .qwen25: return "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        case .addModel: return nil
        }
    }

    var architectureType: ModelArchitectureType {
        switch self {
        case .gemma4: return .vlm
        default: return .llm
        }
    }
}

class PDFParser {
    static let pageBreakMarker = "\u{0}<<<PDF_PAGE_BREAK>>>\u{0}"

    static func cleanFootnoteMarkers(_ text: String) -> String {
        let pattern = "([A-Za-z])(\\d)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return text }
        let nsString = text as NSString
        let fullRange = NSRange(location: 0, length: nsString.length)
        return regex.stringByReplacingMatches(in: text, range: fullRange, withTemplate: "$1 $2")
    }

    static func extractText(from url: URL) -> String? {
        guard let pdf = PDFDocument(url: url) else { return nil }
        var pageTexts: [String] = []
        for i in 0..<pdf.pageCount {
            guard let page = pdf.page(at: i) else { continue }
            let rawText = page.string ?? ""
            pageTexts.append(cleanFootnoteMarkers(rawText))
        }
        return pageTexts.joined(separator: "\n\n\(pageBreakMarker)\n\n")
    }
}

// MARK: - Manual Bridges
private struct ManualHubDownloader: MLXLMCommon.Downloader {
    private let upstream: HuggingFace.HubClient

    init(_ upstream: HuggingFace.HubClient = HuggingFace.HubClient()) {
        self.upstream = upstream
    }

    func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Foundation.Progress) -> Void
    ) async throws -> URL {
        guard let repoID = HuggingFace.Repo.ID(rawValue: id) else {
            throw HuggingFaceDownloaderError.invalidRepositoryID(id)
        }
        let revision = revision ?? "main"

        return try await upstream.downloadSnapshot(
            of: repoID,
            revision: revision,
            matching: patterns,
            progressHandler: { @MainActor progress in
                progressHandler(progress)
            }
        )
    }
}

private struct ManualTokenizerBridge: MLXLMCommon.Tokenizer {
    private var upstream: any Tokenizers.Tokenizer { didSet {} }

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

private struct ManualTokenizerLoader: MLXLMCommon.TokenizerLoader {
    init() {}

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return ManualTokenizerBridge(upstream)
    }
}

enum ParseStatus: Sendable {
    case notApplicable
    case failed
    case parsed(count: Int)
}

struct ChunkMetrics: Sendable {
    var label: String = ""
    var promptTokens: Int = 0
    var generatedTokens: Int = 0
    var prefillSeconds: Double = 0
    var decodeSeconds: Double = 0
    var thinkingChars: Int = 0
    var responseChars: Int = 0
    var hitTokenLimit: Bool = false
    var stopReasonText: String = "unknown"
    var parseStatus: ParseStatus = .notApplicable

    var originalChars: Int = 0
    var sentChars: Int = 0
    var skipped: Bool = false

    var totalSeconds: Double { prefillSeconds + decodeSeconds }

    var thinkingShare: Double {
        let total = thinkingChars + responseChars
        guard total > 0 else { return 0 }
        return Double(thinkingChars) / Double(total)
    }

    var estimatedThinkingTokens: Int {
        Int((Double(generatedTokens) * thinkingShare).rounded())
    }

    var sentShare: Double {
        guard originalChars > 0 else { return 1 }
        return Double(sentChars) / Double(originalChars)
    }

    var summaryLine: String {
        if skipped {
            return "Batch \(label) | SKIPPED (no citation-shaped text in \(originalChars) chars)"
        }
        var parts: [String] = ["Batch \(label)"]
        parts.append(String(format: "%.2fs (prefill %.2fs / decode %.2fs)",
                            totalSeconds, prefillSeconds, decodeSeconds))
        if originalChars > 0, sentChars < originalChars {
            parts.append(String(format: "sent %d/%d chars (%.0f%%)",
                                sentChars, originalChars, sentShare * 100))
        }
        parts.append("\(promptTokens) prompt → \(generatedTokens) generated tok")
        if thinkingChars > 0 {
            parts.append(String(format: "thinking ~%d tok (%.0f%% of output)",
                                estimatedThinkingTokens, thinkingShare * 100))
        }
        switch parseStatus {
        case .parsed(let count): parts.append("\(count) cases")
        case .failed: parts.append("⚠️ JSON PARSE FAILED")
        case .notApplicable: break
        }
        if hitTokenLimit {
            parts.append("⚠️ TRUNCATED (hit maxTokens)")
        } else if stopReasonText == "cancelled" {
            parts.append("⚠️ cancelled")
        }
        return parts.joined(separator: " | ")
    }
}

struct ChunkResult: Sendable {
    let text: String
    let metrics: ChunkMetrics
    let thinkingText: String
}

// MARK: - Citation Scanner
enum CitationScanner {
    static let defaultCharsBefore = 350
    static let defaultCharsAfter = 350
    static let mergeGap = 150

    private static let reporterTokens: [String] = [
        "MANU", "MANUPATRA", "MhLJ", "Mh.L.J.", "SCR", "S.C.R.",
        "ELT", "AIR", "ITR", "SCC", "SSC", "CRLJ", "CTR", "TAXMAN", "LLJ",
        "DLT", "Scale", "RLT", "STC", "SCR", "JT", "TTJ", "ITD",
        "ECR", "CPJ", "CLR", "AD", "Supreme", "DRJ", "SLT", "SLR",
        "STR", "ARBLR", "SCJ", "AllMR", "UJ", "FLR", "SRJ", "Crimes",
        "KarLJ", "LLN", "CCC", "KLT", "LIC", "CCR", "CTJ", "SLJ",
        "TLR", "PTC", "ATC", "CTC", "RCR", "CPR", "FJR", "VST",
        "MLJ", "JCC", "RecentCR", "CLT", "ALT", "SLJCAT", "CLJ", "RentLR",
        "RCJ", "FAC", "DRTC", "SOT", "CRJ", "ECC", "CompLJ", "CLA",
        "CLC", "SCW", "CCrC", "CHN", "DMC", "ACC", "ILR", "ALLMR",
        "KHCACJ", "HLR", "ATR", "ETR", "AIC", "BCR", "OCR", "ITJ",
        "MTJ", "TAXATION", "ALR", "BankCLR", "MPLJ", "ACJ", "BLJR", "ALJ",
        "AnWR", "RRR", "GLR", "STT", "PunjLR", "AnLT", "GLH", "MhLJ",
        "MahLJ", "AllER", "CutLT", "KerLR", "LILR", "BLR", "VKN", "KLJ",
        "AllCriC", "SCL", "GujLR", "PLR", "JKLR", "GCD", "PLJR", "RLR",
        "GujLH", "OELT", "BLJ", "BomLR", "KerLJ", "MIA", "SCt", "UPLBEC",
        "SarPCJ", "ACE", "WLR", "CalLT", "MWN", "TAC", "EWHC", "UKSC"
    ]

    private static let neutralTokens: [String] = [
        "Civil Appeal", "C.A.", "Criminal Appeal", "Crl.A.", "Company Appeal",
        "APHC", "KER", "KHC", "RJ-JD", "CGHC", "INSC", "MPHC-JBP", "GUJHC",
        "HHC", "MPHC-IND", "MPHC-GWL", "RJ-JP", "DHC", "BHC-NAG", "MHC", "BHC-AUG",
        "PHHC", "AHC", "GAU-AS", "KHC-D", "MLHC", "KHC-K", "AHC-LKO", "BHC-OS",
        "BHC-GOA", "CHC-AS", "SHC", "THC", "CHC-OS", "JHHC", "BHC-KOL", "CHC-PB",
        "BHC-AS", "JKLHC-JMU", "UHC", "JKLHC-SGR", "OHC", "CHC-JP"
    ]

    private static let docketTokens: [String] = [
        "SLP(C)", "SLP(Crl)", "SLP", "W.P.(C)", "WP(C)", "W.P.(Crl)", "WP(Crl)", "W.P.", "WP",
        "CS(COMM)", "CS(OS)", "CS", "FAO", "RFA", "CRL.A.", "Crl.A.", "O.M.P.", "ARB.P.",
        "MAT.APP.", "CONT.CAS", "CRL.M.C.", "CM APPL."
    ]

    static var includePartyMarker = true

    private static let citationRegex: NSRegularExpression = {
        func tolerantAcronym(_ token: String) -> String {
            let escaped = NSRegularExpression.escapedPattern(for: token)
            guard token.allSatisfy({ $0.isLetter || $0 == "&" || $0 == " " || $0 == "-" }) else {
                return "(?i:\(escaped))"
            }
            let pattern = token.map { char -> String in
                if char == " " { return "\\s+" }
                return "\(NSRegularExpression.escapedPattern(for: String(char)))\\.?"
            }.joined()
            return "(?i:\(pattern))"
        }

        let standardTokens = (reporterTokens + neutralTokens)
            .sorted { ($0.count, $0) > ($1.count, $1) }
            .map(tolerantAcronym)
            .joined(separator: "|")

        let docketEscaped = docketTokens
            .sorted { ($0.count, $0) > ($1.count, $1) }
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
        let docketPattern = "(?i:(?:\(docketEscaped)))\\s*(?:\\(?[A-Za-z]+\\)?\\s*)?(?:No\\.?|Nos\\.?)?\\s*\\d+"

        let sccOnlinePattern = "(?i:SCC\\s*OnLine(?:\\s*[A-Za-z]{2,6})?\\s*\\d+)"

        var combinedPatterns: [String] = [
            "(?<![A-Za-z])(?:\(standardTokens))(?:\\s*\\([A-Za-z&.\\- ]{1,14}\\))?(?![A-Za-z])",
            sccOnlinePattern,
            docketPattern
        ]

        if includePartyMarker {
            let partyPattern = "(?<=[\\w\\)\\]\\.'’\"][\\s\\-,])(?i:v|vs|v/s)\\.?(?=[\\s\\-,;:]|$)|(?i:\\bversus\\b)|(?i:\\bIn\\s+Re:?\\b)|(?i:\\btitled\\s+['\"‘][A-Za-z])"
            combinedPatterns.append(partyPattern)
        }

        let fullPattern = combinedPatterns.joined(separator: "|")
        return try! NSRegularExpression(pattern: fullPattern, options: [])
    }()

    private static func isPureHeaderPassage(_ text: String) -> Bool {
        guard text.count < 350 else { return false }
        let upper = text.uppercased()
        let hasHeaderMarker = (upper.contains("APPELLANT") && upper.contains("RESPONDENT")) ||
                              (upper.contains("PETITIONER") && upper.contains("RESPONDENT")) ||
                              (upper.contains("IN THE SUPREME COURT") || upper.contains("IN THE HIGH COURT"))

        let hasJudicialVerb = upper.contains("HELD") || upper.contains("RELIED") || upper.contains("OBSERVED") || upper.contains("CITED") || upper.contains("FOLLOWED")
        return hasHeaderMarker && !hasJudicialVerb
    }

    static func containsCitation(_ text: String) -> Bool {
        let ns = text as NSString
        return citationRegex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) != nil
    }

    private static func snapToWordBoundaries(in text: NSString, range: NSRange) -> NSRange {
        var start = range.location
        var end = range.location + range.length
        
        if start > 0 {
            var searchIdx = start
            while searchIdx > 0 && searchIdx > start - 40 {
                let char = text.character(at: searchIdx)
                if char == 10 || char == 32 {
                    start = searchIdx + 1
                    break
                }
                searchIdx -= 1
            }
        }
        
        if end < text.length {
            var searchIdx = end
            while searchIdx < text.length && searchIdx < end + 40 {
                let char = text.character(at: searchIdx)
                if char == 10 || char == 32 {
                    end = searchIdx
                    break
                }
                searchIdx -= 1
            }
        }
        
        return NSRange(location: start, length: max(0, end - start))
    }

    struct AnnotatedPassage {
        let text: String
        let pageNumber: Int
        let endPageNumber: Int

        init(text: String, pageNumber: Int, endPageNumber: Int? = nil) {
            self.text = text
            self.pageNumber = pageNumber
            self.endPageNumber = endPageNumber ?? pageNumber
        }

        var spansPages: Bool { endPageNumber != pageNumber }
    }

    private static func pageNumberAt(offset: Int, in text: NSString, pageMarkerPositions: [(offset: Int, page: Int)]) -> Int {
        var bestPage = pageMarkerPositions.first?.page ?? 1
        for (markerOffset, page) in pageMarkerPositions {
            if markerOffset <= offset {
                bestPage = page
            } else {
                break
            }
        }
        return bestPage
    }

    private static func findPageMarkers(in text: NSString) -> [(offset: Int, page: Int)] {
        let markerPrefix = LLMManager.internalPageMarker
        let markerSuffix = LLMManager.internalPageMarkerSuffix
        var markers: [(offset: Int, page: Int)] = []
        var searchStart = 0
        while searchStart < text.length {
            let range = text.range(of: markerPrefix, options: [], range: NSRange(location: searchStart, length: text.length - searchStart))
            guard range.location != NSNotFound else { break }
            let afterPrefix = range.location + range.length
            let suffixRange = text.range(of: markerSuffix, options: [], range: NSRange(location: afterPrefix, length: min(10, text.length - afterPrefix)))
            if suffixRange.location != NSNotFound {
                let pageStr = text.substring(with: NSRange(location: afterPrefix, length: suffixRange.location - afterPrefix))
                if let pageNum = Int(pageStr) {
                    markers.append((offset: range.location, page: pageNum))
                }
            }
            searchStart = range.location + range.length
        }
        return markers
    }

    static func candidatePassages(
        in text: String,
        charsBefore: Int = defaultCharsBefore,
        charsAfter: Int = defaultCharsAfter
    ) -> [AnnotatedPassage]? {
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        let matches = citationRegex.matches(in: text, range: full)
        guard !matches.isEmpty else { return nil }

        let pageMarkers = findPageMarkers(in: ns)

        var rawEntries: [(range: NSRange, page: Int)] = []
        for match in matches {
            let start = max(0, match.range.location - charsBefore)
            let end = min(ns.length, match.range.location + match.range.length + charsAfter)
            let page = pageNumberAt(offset: match.range.location, in: ns, pageMarkerPositions: pageMarkers)
            rawEntries.append((range: NSRange(location: start, length: end - start), page: page))
        }

        let sorted = rawEntries.sorted { $0.range.location < $1.range.location }
        var merged: [(range: NSRange, page: Int)] = []
        for entry in sorted {
            if let last = merged.last {
                let lastEnd = last.range.location + last.range.length
                if entry.range.location <= lastEnd + mergeGap {
                    merged[merged.count - 1].range.length = max(lastEnd, entry.range.location + entry.range.length) - last.range.location
                } else {
                    merged.append(entry)
                }
            } else {
                merged.append(entry)
            }
        }

        var passages: [AnnotatedPassage] = merged.compactMap { entry in
            let snapped = snapToWordBoundaries(in: ns, range: entry.range)
            guard snapped.length > 0 else { return nil }
            var passage = ns.substring(with: snapped).trimmingCharacters(in: .whitespacesAndNewlines)
            if isPureHeaderPassage(passage) { return nil }
            if passage.isEmpty { return nil }

            let lastOffset = max(snapped.location, snapped.location + snapped.length - 1)
            let startPage = pageNumberAt(offset: snapped.location, in: ns, pageMarkerPositions: pageMarkers)
            let endPage = pageNumberAt(offset: lastOffset, in: ns, pageMarkerPositions: pageMarkers)

            let markerPattern = "\\n*<<<PAGE_(\\d+)>>>\\n*"
            if let regex = try? NSRegularExpression(pattern: markerPattern) {
                let ns = passage as NSString
                var rebuilt = ""
                var cursor = 0
                for m in regex.matches(in: passage, range: NSRange(location: 0, length: ns.length)) {
                    rebuilt += ns.substring(with: NSRange(location: cursor, length: m.range.location - cursor))
                    let page = ns.substring(with: m.range(at: 1))
                    let tail = rebuilt.trimmingCharacters(in: .whitespacesAndNewlines)
                    var sentenceEnded = true
                    if let lastChar = tail.last {
                        if ".!?:;\"”'’".contains(lastChar) {
                            let lastToken = tail.split(separator: " ").last.map(String.init) ?? ""
                            sentenceEnded = !(lastChar == "." && lastToken.count <= 4)
                        } else {
                            sentenceEnded = false
                        }
                    }
                    rebuilt += sentenceEnded ? "\n\n[Page \(page)]\n" : " [Page \(page)] "
                    cursor = m.range.location + m.range.length
                }
                rebuilt += ns.substring(from: cursor)
                passage = rebuilt
            }
            passage = passage.trimmingCharacters(in: .whitespacesAndNewlines)

            let redundantPrefix = "[Page \(startPage)]"
            if passage.hasPrefix(redundantPrefix) {
                passage = String(passage.dropFirst(redundantPrefix.count))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard !passage.isEmpty else { return nil }

            return AnnotatedPassage(text: passage, pageNumber: startPage, endPageNumber: endPage)
        }

        guard !passages.isEmpty else { return nil }

        let lines = text.components(separatedBy: .newlines)
        let footnoteLines = lines.filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let firstChar = trimmed.first, firstChar.isNumber else { return false }
            return trimmed.contains("SCC") || trimmed.contains("AIR") || trimmed.contains("INSC") || trimmed.contains("Supp") || trimmed.contains("Judges") || trimmed.contains("EWHC")
        }

        if !footnoteLines.isEmpty {
            let lastPage = passages.last?.pageNumber ?? pageMarkers.last?.page ?? 1
            let footnotesBlock = "FOOTNOTES FOR THIS PAGE:\n" + footnoteLines.joined(separator: "\n")
            passages.append(AnnotatedPassage(text: footnotesBlock, pageNumber: lastPage))
        }

        return passages
    }

    static func formatAnnotatedPassages(_ passages: [AnnotatedPassage]) -> String {
        passages.map { passage in
            let header = passage.spansPages
                ? "[Page \(passage.pageNumber)-\(passage.endPageNumber) excerpt]"
                : "[Page \(passage.pageNumber) excerpt]"
            return "\(header):\n\(passage.text)"
        }
        .joined(separator: "\n\n[…]\n\n")
    }
}

// MARK: - Stream Parser
class StreamingReasoningParser {
    enum State {
        case preThinking
        case thinking
        case normal
    }
    
    private var state: State = .preThinking
    private var buffer = ""
    private let thinkingEnabled: Bool
    
    let onThinking: (String) -> Void
    let onNormal: (String) -> Void
    
    init(thinkingEnabled: Bool, onThinking: @escaping (String) -> Void, onNormal: @escaping (String) -> Void) {
        self.thinkingEnabled = thinkingEnabled
        self.onThinking = onThinking
        self.onNormal = onNormal
        
        if !thinkingEnabled {
            self.state = .normal
        }
    }
    
    func process(_ token: String) {
        buffer += token

        while !buffer.isEmpty {
            switch state {
            case .preThinking:
                if let startRange = findStartDelimiter(in: buffer) {
                    state = .thinking
                    buffer = String(buffer[startRange.upperBound...])
                } else if buffer.count > 300 {
                    state = .normal
                    onNormal(buffer)
                    buffer = ""
                    return
                } else {
                    return
                }
                
            case .thinking:
                if let endRange = findEndDelimiter(in: buffer) {
                    let thinkingText = String(buffer[..<endRange.lowerBound])
                    if !thinkingText.isEmpty {
                        onThinking(thinkingText)
                    }
                    state = .normal
                    buffer = String(buffer[endRange.upperBound...])
                } else {
                    let safetyMargin = 20
                    if buffer.count > safetyMargin {
                        let sendText = String(buffer.prefix(buffer.count - safetyMargin))
                        onThinking(sendText)
                        buffer = String(buffer.suffix(safetyMargin))
                    }
                    return
                }
                
            case .normal:
                onNormal(buffer)
                buffer = ""
            }
        }
    }
    
    func finalize() {
        if !buffer.isEmpty {
            switch state {
            case .preThinking:
                onNormal(buffer)
            case .thinking:
                onThinking(buffer)
            case .normal:
                onNormal(buffer)
            }
        }
        buffer = ""
    }
    
    private func findStartDelimiter(in text: String) -> Range<String.Index>? {
        let delimiters = [
            "<|channel>thought",
            "<|thought|>",
            "<|think|>",
            "<think>"
        ]
        for delim in delimiters {
            if let range = text.range(of: delim, options: .caseInsensitive) {
                return range
            }
        }
        return nil
    }
    
    private func findEndDelimiter(in text: String) -> Range<String.Index>? {
        let delimiters = [
            "<channel|>",
            "<|/thought|>",
            "<|/think|>",
            "</think>"
        ]
        for delim in delimiters {
            if let range = text.range(of: delim, options: .caseInsensitive) {
                return range
            }
        }
        return nil
    }
}

// MARK: - LLM Manager
@MainActor
class LLMManager: ObservableObject {
    @Published var isGenerating = false
    @Published var downloadProgress: Double = 0.0
    @Published var statusText: String = ""
    @Published var generationTimeText: String = ""

    let jsonSystemPrompt = """
    You are an expert Legal AI Assistant specializing in Indian jurisprudence and legal document analysis.
    Your task is to carefully read through Indian court filing documents and extract EVERY legal case named in them — whether relied on as authority, distinguished, overruled, quoted, or merely discussed while checking whether a citation is genuine.

    Follow these strict guidelines:
    1. Identify Case Names (STRICT SEPARATION):
       - Look for standard adversarial formats (e.g., "X v. Y", "X vs. Y", "In Re: X"). Clean out trailing footnote numbers.
       - The "caseName" field MUST ONLY contain the names of the parties. NEVER include the citation, volume, year, court name, or page numbers inside caseName.

    2. Identify Citations (EXACT VERBATIM & STRICT SEPARATION): 
       - The "citation" field MUST ONLY contain the reporter reference (e.g., "2010 (251) ELT 348 (Mad.)", "(1996) 10 SCC 413", "2025 SCC OnLineBom 1262").
       - NEVER include party names, "v.", or bullet symbols inside the "citation" field.
       - Always copy the EXACT characters, brackets, and spelling as they appear in the original text without moving parentheses.

    3. BULLETED & NUMBERED LISTS (CRITICAL):
       - Court judgments often list cited authorities in bullet points (•) or numbered lists (1, 2, 3...).
       - You MUST extract EVERY SINGLE bullet point without exception. If there are 3 consecutive bullets, you MUST output all 3 cases. Do NOT stop after extracting only 1 or 2.

    4. RECENT PRECEDENTS (2024 / 2025 / 2026):
       - If a cited case has a recent citation (such as "2025 SCC OnLineBom 1262" or "2024 INSC 500"), it IS a genuine cited precedent and MUST be extracted.
       - Only exclude the current document's own filing number (e.g., W.P.(C) 16754/2025). Do NOT ignore cited authorities just because they are from recent years.

    5. Context: Briefly summarize the legal principle or reason why the case was cited (if any).
    6. Missing Data: If a detail is missing, return null. Do not hallucinate.
    7. Repeats: If the same case is cited more than once for different points, emit it once per occurrence, each with its own context and page number. Do not merge them.
    
    8. FOOTNOTES & SUPERSCRIPTS (CRITICAL):
       - Numbers immediately following a case name (e.g. "Kesavananda Bharati v. State of Kerala¹" or "S R Bommai vs Union of India 2") are FOOTNOTE MARKERS, NOT citations.
       - NEVER output a bare number (such as "1", "2", "3", "4") in the "citation" field.
       - Whenever a case refers to a footnote marker, check the "FOOTNOTES FOR THIS PAGE" block at the bottom of the excerpt.
       - Look up the matching footnote number (e.g. "2 (1994) 3 SCC 1 (9 Judges)") and use the real legal reporter citation in the "citation" field, and set the "year".

    9. Page Number: Set pageNumber from the page annotations in the text.
       - "[Page N excerpt]" means everything under it is on page N.
       - "[Page A-B excerpt]" means the passage crosses a page break. Inside it, an inline "[Page B]" marks exactly where page B begins.
       - A case name split across the break belongs to the page where the CASE NAME starts.
       - If genuinely ambiguous, use the first page number visible.

    10. Continuous Text: A page marker NEVER splits a case. Text runs continuously across it, so join the halves into a single entry. Never take a page marker, page number or filing number as a precedent's citation.

    OUTPUT FORMAT:
    Respond with ONE JSON object and nothing else. Do NOT write markdown fences.
    Do NOT write type declarations. Do NOT explain.
    Every key and every string value MUST be wrapped in double quotes.

    Return exactly this shape:
    {"citedCases":[{"caseName":"Kesavananda Bharati v. State of Kerala","citation":"(1973) 4 SCC 225","year":1973,"court":"Supreme Court of India","context":"Observed secularism is a basic feature of the Constitution.","pageNumber":4},{"caseName":"S.R. Bommai v. Union of India","citation":"(1994) 3 SCC 1","year":1994,"court":"Supreme Court of India","context":"Cited on secularism as basic structure.","pageNumber":4}]}

    Field rules:
    - "caseName": string, required (without trailing footnote numbers)
    - "citation": string or null (never a bare digit like 1 or 2; must be a real reporter citation or null)
    - "year": number or null
    - "court": string or null
    - "context": string or null
    - "pageNumber": number or null

    If no cases are cited, return exactly: {"citedCases":[]}
    """

    let passageExcerptNote = """

    INPUT FORMAT NOTE:
    The text below consists of candidate excerpts surrounding cited cases. Each excerpt is prefixed with [Page N excerpt] indicating the PDF page number. Footnotes appear at the bottom under "FOOTNOTES FOR THIS PAGE:". Map any footnote numbers in case names to their actual legal citations listed in the footnotes block.
    """

    let strictThinkingDirective = """

    REASONING GUIDELINE (Inside <|channel>thought):
    - Write your thoughts as a natural, conversational stream-of-consciousness (thinking out loud).
    - Avoid robotic templates, repeating lines, "1. Analyze", "Step 1", or structured tables.
    - Start directly with conversational phrases like: "Scanning this chunk...", "Footnote 2 maps to (1994) 3 SCC 1..."
    - Keep thoughts fluid, conversational, and under 50 words total. Do not write "Thinking Process:" or "Thinking:" as the UI displays it.
    """
    
    private var loadedContainers: [String: ModelContainer] = [:]

    private func loadContainer(repoID: String, architectureType: ModelArchitectureType) async throws -> ModelContainer {
        if let cached = loadedContainers[repoID] {
            return cached
        }

        downloadProgress = 0.0
        statusText = "Checking model repository..."
        let configuration = ModelConfiguration(id: repoID)

        let progressClosure: @Sendable (Progress) -> Void = { progress in
            Task { @MainActor in
                self.downloadProgress = progress.fractionCompleted
                let completedMB = Double(progress.completedUnitCount) / (1024 * 1024)
                let totalMB = Double(progress.totalUnitCount) / (1024 * 1024)
                let pct = Int(progress.fractionCompleted * 100)
                
                if progress.totalUnitCount > 0 && totalMB > 1.0 {
                    if totalMB >= 1024 {
                        let completedGB = completedMB / 1024.0
                        let totalGB = totalMB / 1024.0
                        self.statusText = String(format: "Downloading: %.2f GB / %.2f GB (%d%%)", completedGB, totalGB, pct)
                    } else {
                        self.statusText = String(format: "Downloading: %.1f MB / %.1f MB (%d%%)", completedMB, totalMB, pct)
                    }
                } else if progress.fractionCompleted > 0 {
                    self.statusText = "Downloading model: \(pct)%"
                } else {
                    self.statusText = "Connecting & downloading model..."
                }
                
                if progress.fractionCompleted >= 1.0 {
                    self.statusText = "Loading model into memory..."
                }
            }
        }

        let container: ModelContainer
        switch architectureType {
        case .llm:
            container = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: progressClosure
            )
        case .vlm:
            container = try await VLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: progressClosure
            )
        }
        
        statusText = "Warming up GPU engine..."
        await Self.warmup(container: container)

        loadedContainers[repoID] = container
        downloadProgress = 1.0
        statusText = ""
        return container
    }

    private static func warmup(container: ModelContainer) async {
        _ = try? await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(prompt: .text("Extract cited legal precedents and cases from the provided document excerpts."))
            )
            var params = GenerateParameters(temperature: 0.0)
            params.maxTokens = 1

            let stream = try MLXLMCommon.generate(input: input, parameters: params, context: context)
            for await _ in stream {
                break
            }
        }
    }

    func generateStructuredOutput(
        pdfText: String,
        model: LLMModel,
        thinkingEnabled: Bool = true,
        preFilterEnabled: Bool = true,
        onThinkingToken: @MainActor @escaping (String) -> Void,
        onToken: @MainActor @escaping (String) -> Void
    ) async throws -> GenerationResult {
        guard let repoID = model.hubRepoID else {
            throw LLMError.noRepoConfigured
        }

        isGenerating = true
        downloadProgress = 0.0
        generationTimeText = "Processing..."
        defer {
            isGenerating = false
            downloadProgress = 0.0
        }

        let container = try await loadContainer(repoID: repoID, architectureType: model.architectureType)
        let basePrompt = jsonSystemPrompt
        let promptWithThinking = thinkingEnabled ? (basePrompt + strictThinkingDirective) : basePrompt

        let totalStartTime = CFAbsoluteTimeGetCurrent()

        let cleanedText = Self.stripRunningHeaders(pdfText)
        let batches = Self.chunkByPages(cleanedText)
        let isSingleBatch = batches.count == 1

        let pageTexts = cleanedText.components(separatedBy: PDFParser.pageBreakMarker)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        statusText = "Identifying document's own case..."
        let selfRef = try await extractSelfReference(
            container: container,
            firstBatchText: batches.first?.text ?? "",
            thinkingEnabled: thinkingEnabled
        )

        var exclusionClause = ""
        if let title = selfRef.caseTitle {
            exclusionClause += "\n\n    SELF-REFERENCE EXCLUSION: This document's own case is \"\(title)\""
            if let number = selfRef.caseNumber {
                exclusionClause += " (Case No. \(number))"
            }
            exclusionClause += ". Do NOT include this case in your output — it is the document itself, not a cited precedent."
        }

        let promptWithExclusion = promptWithThinking + exclusionClause
        let activePrompt = preFilterEnabled ? (promptWithExclusion + passageExcerptNote) : promptWithExclusion

        var chunkOutputs: [String] = []
        var chunkThinking: [String] = []
        var allMetrics: [ChunkMetrics] = []

        onToken("")
        onThinkingToken("")

        for (index, batch) in batches.enumerated() {
            let batchLabel = "\(index + 1)/\(batches.count) \(batch.pageRangeLabel)"

            statusText = isSingleBatch
                ? "Processing document..."
                : "Processing batch \(index + 1) of \(batches.count) (\(batch.pageRangeLabel))..."

            var textToSend = batch.text
            if preFilterEnabled {
                guard let passages = CitationScanner.candidatePassages(in: batch.text) else {
                    var skippedMetrics = ChunkMetrics()
                    skippedMetrics.label = batchLabel
                    skippedMetrics.originalChars = batch.text.count
                    skippedMetrics.skipped = true
                    allMetrics.append(skippedMetrics)
                    print("⏭️  [SKIP] \(skippedMetrics.summaryLine)")
                    continue
                }
                textToSend = CitationScanner.formatAnnotatedPassages(passages)
            } else {
                let markerPattern = "<<<PAGE_(\\d+)>>>\\n?"
                if let regex = try? NSRegularExpression(pattern: markerPattern) {
                    textToSend = regex.stringByReplacingMatches(
                        in: textToSend,
                        range: NSRange(location: 0, length: (textToSend as NSString).length),
                        withTemplate: "[Page $1]:\n"
                    )
                }
            }

            let batchSource = preFilterEnabled
                ? "excerpts taken from pages \(batch.pageRangeLabel)"
                : "a batch of pages (\(batch.pageRangeLabel))"
            let label: String? = isSingleBatch
                ? nil
                : "This is \(batchSource) of a larger legal document, part \(index + 1) of \(batches.count) overall. Extract cited cases found in THIS BATCH only. Use the [Page N excerpt] or [Page N] markers to set each case's pageNumber field."

            print("▶️ [START] Batch \(index + 1)/\(batches.count) (\(batch.pageRangeLabel))...")

            let result = try await generateForChunk(
                container: container,
                activePrompt: activePrompt,
                chunkText: textToSend,
                chunkLabel: label,
                thinkingEnabled: thinkingEnabled,
                onThinkingToken: onThinkingToken,
                onToken: onToken
            )

            var metrics = result.metrics
            metrics.label = batchLabel
            metrics.originalChars = batch.text.count
            metrics.sentChars = textToSend.count

            if let parsed = DocumentExtractionResult.parse(from: result.text) {
                metrics.parseStatus = .parsed(count: parsed.citedCases.count)
            } else {
                metrics.parseStatus = .failed
            }

            chunkOutputs.append(result.text)
            chunkThinking.append(result.thinkingText)
            allMetrics.append(metrics)

            print("✅ [COMPLETE] \(metrics.summaryLine)")
        }

        var mergedResult: String
        if chunkOutputs.isEmpty {
            mergedResult = "{\"citedCases\":[]}"
        } else if isSingleBatch {
            mergedResult = Self.cleanedIfPossible(chunkOutputs[0])
        } else {
            statusText = "Merging results from \(chunkOutputs.count) page-batches..."
            mergedResult = Self.mergeChunkOutputs(chunkOutputs)
        }

        if let parsed = DocumentExtractionResult.parse(from: mergedResult) {
            var finalCases = Self.filterSelfReferences(parsed.citedCases, selfRef: selfRef)
            if finalCases.count < parsed.citedCases.count {
                print("🧹 [SELF-REF FILTER] Removed \(parsed.citedCases.count - finalCases.count) duplicate self-reference(s) from the cited list")
            }
            
            // 1. Resolve Correct Page Numbers
            finalCases = Self.resolvePageNumbers(finalCases, pageTexts: pageTexts)
            
            // 2. Resolve Footnotes
            finalCases = Self.resolveFootnoteCitations(finalCases, pageTexts: pageTexts)
            
            // 3. Resolve Exact Citations & Case Names from PDF for Highlighting
            finalCases = Self.resolveExactPDFCitation(finalCases, pageTexts: pageTexts)
            
            if let selfRow = Self.makeSelfReferenceRow(selfRef) {
                finalCases.insert(selfRow, at: 0)
            }
            let result = DocumentExtractionResult(citedCases: finalCases)
            if let data = try? JSONEncoder().encode(result), let json = String(data: data, encoding: .utf8) {
                mergedResult = json
            }
        }

        let finalResult = mergedResult
        let finalThinking = chunkThinking.joined(separator: "\n\n")

        let totalTimeTaken = CFAbsoluteTimeGetCurrent() - totalStartTime
        Self.printRunSummary(allMetrics, totalSeconds: totalTimeTaken)

        let formattedTime = String(format: "%.2f", totalTimeTaken)
        let totalDecodeSeconds = allMetrics.reduce(0.0) { $0 + $1.decodeSeconds }
        let totalGeneratedTokens = allMetrics.reduce(0) { $0 + $1.generatedTokens }
        if totalDecodeSeconds > 0 {
            let rate = Double(totalGeneratedTokens) / totalDecodeSeconds
            self.generationTimeText = String(format: "Time taken: %@ seconds (%.1f tok/s)", formattedTime, rate)
        } else {
            self.generationTimeText = "Time taken: \(formattedTime) seconds"
        }
        self.statusText = "Done!"

        return GenerationResult(responseText: finalResult, thinkingText: finalThinking)
    }

    private struct PageBatch {
        let text: String
        let pageRangeLabel: String
        let startPageNumber: Int
        let endPageNumber: Int
    }

    private let selfReferencePrompt = """
    You are reading the FIRST TWO PAGES of an Indian court document (judgment, order, or petition).
    Your ONLY task: identify the document's OWN case title and case/citation number.
    This is NOT about cited precedents — it is about the document itself.

    Look for:
    - The title block (e.g., "Party A v. Party B" or "In the matter of XYZ")
    - The case/filing number (e.g., "SLP(C) No. 1234/2025", "W.P.(C) 5678/2024", "2025:INSC:789")

    Return ONLY a valid JSON object, no markdown fences:
    {"caseTitle": "Party A v. Party B", "caseNumber": "SLP(C) No. 1234/2025"}
    If you cannot determine a field, set it to null.
    """

    private func extractSelfReference(
        container: ModelContainer,
        firstBatchText: String,
        thinkingEnabled: Bool
    ) async throws -> DocumentSelfReference {
        let result = try await generateForChunk(
            container: container,
            activePrompt: selfReferencePrompt,
            chunkText: firstBatchText,
            chunkLabel: "Identify this document's own case title and citation/case number.",
            thinkingEnabled: thinkingEnabled,
            onThinkingToken: { _ in },
            onToken: { _ in }
        )
        let selfRef = DocumentSelfReference.parse(from: result.text)
        if let title = selfRef.caseTitle {
            print("📋 [SELF-REF] Document title: \(title) | Number: \(selfRef.caseNumber ?? "N/A")")
        } else {
            print("📋 [SELF-REF] Could not identify document's own case title")
        }
        return selfRef
    }

    private static func caseNumberKeys(_ raw: String?) -> Set<String> {
        guard let raw, !raw.isEmpty else { return [] }
        let ns = raw as NSString
        let full = NSRange(location: 0, length: ns.length)
        var keys = Set<String>()

        if let re = try? NSRegularExpression(pattern: "(\\d{1,7})\\s*(?:/|\\s+of\\s+)\\s*((?:19|20)\\d{2})") {
            for m in re.matches(in: raw, range: full) where m.numberOfRanges == 3 {
                keys.insert("\(ns.substring(with: m.range(at: 1)))/\(ns.substring(with: m.range(at: 2)))")
            }
        }

        if let re = try? NSRegularExpression(pattern: "((?:19|20)\\d{2})\\s*[:\\s]\\s*([A-Za-z]{2,10})\\s*[:\\s]\\s*(\\d{1,6})") {
            for m in re.matches(in: raw, range: full) where m.numberOfRanges == 4 {
                let year = ns.substring(with: m.range(at: 1))
                let court = ns.substring(with: m.range(at: 2)).uppercased()
                let num = ns.substring(with: m.range(at: 3))
                keys.insert("\(year):\(court):\(num)")
            }
        }

        return keys
    }

    private static func makeSelfReferenceRow(_ selfRef: DocumentSelfReference) -> CitedCase? {
        guard let name = selfRef.caseTitle ?? selfRef.caseNumber else { return nil }
        let citation = (selfRef.caseTitle != nil) ? selfRef.caseNumber : nil

        return CitedCase(
            caseName: name,
            citation: citation,
            year: nil,
            court: nil,
            context: "This document's own case — not a cited precedent.",
            pageNumber: 1,
            isSelfReference: true
        )
    }

    private static func filterSelfReferences(_ cases: [CitedCase], selfRef: DocumentSelfReference) -> [CitedCase] {
        let selfTokens = selfRef.caseTitle.map(nameTokens) ?? []
        let selfNumberKeys = caseNumberKeys(selfRef.caseNumber)

        guard !selfTokens.isEmpty || !selfNumberKeys.isEmpty else { return cases }

        return cases.filter { c in
            if !selfNumberKeys.isEmpty {
                let candidateKeys = caseNumberKeys(c.citation).union(caseNumberKeys(c.caseName))
                if !candidateKeys.isDisjoint(with: selfNumberKeys) { return false }
            }

            guard !selfTokens.isEmpty else { return true }
            let caseTokens = nameTokens(c.caseName)
            guard !caseTokens.isEmpty else { return true }

            let intersection = selfTokens.intersection(caseTokens)
            let jaccard = Double(intersection.count) / Double(selfTokens.union(caseTokens).count)
            if jaccard > 0.7 { return false }

            let minCount = min(selfTokens.count, caseTokens.count)
            if minCount >= 2 && intersection.count == minCount { return false }

            return true
        }
    }

    private static func printRunSummary(_ metrics: [ChunkMetrics], totalSeconds: Double) {
        guard !metrics.isEmpty else { return }

        let prefill = metrics.reduce(0.0) { $0 + $1.prefillSeconds }
        let decode = metrics.reduce(0.0) { $0 + $1.decodeSeconds }
        let measured = prefill + decode
        let promptTokens = metrics.reduce(0) { $0 + $1.promptTokens }
        let generatedTokens = metrics.reduce(0) { $0 + $1.generatedTokens }
        let thinkingChars = metrics.reduce(0) { $0 + $1.thinkingChars }
        let responseChars = metrics.reduce(0) { $0 + $1.responseChars }
        let estThinkingTokens = metrics.reduce(0) { $0 + $1.estimatedThinkingTokens }
        let truncated = metrics.filter { $0.hitTokenLimit }
        let skipped = metrics.filter { $0.skipped }
        let originalChars = metrics.reduce(0) { $0 + $1.originalChars }
        let sentChars = metrics.reduce(0) { $0 + $1.sentChars }
        let parseFailures = metrics.filter {
            if case .failed = $0.parseStatus { return true }
            return false
        }

        func pct(_ part: Double, of whole: Double) -> String {
            guard whole > 0 else { return "n/a" }
            return String(format: "%.0f%%", part / whole * 100)
        }

        let decodeRate = decode > 0
            ? String(format: "%.1f tok/s", Double(generatedTokens) / decode)
            : "n/a"
        let thinkingPct = pct(Double(thinkingChars), of: Double(thinkingChars + responseChars))

        let filterLine = originalChars > 0
            ? "\(sentChars)/\(originalChars) chars sent (\(pct(Double(sentChars), of: Double(originalChars)))), \(skipped.count) batch(es) skipped"
            : "not applied"

        print("""

        ═══════════ RUN SUMMARY (JSON) ═══════════
        Batches:        \(metrics.count) (\(metrics.count - skipped.count) sent to the model)
        Pre-filter:     \(filterLine)
        Wall clock:     \(String(format: "%.2fs", totalSeconds))  (in-model: \(String(format: "%.2fs", measured)))
        Prefill:        \(String(format: "%.2fs", prefill))  (\(pct(prefill, of: measured)) of model time)  \(promptTokens) tok
        Decode:         \(String(format: "%.2fs", decode))  (\(pct(decode, of: measured)) of model time)  \(generatedTokens) tok
        Decode rate:    \(decodeRate)
        Thinking:       ~\(estThinkingTokens) of \(generatedTokens) generated tok (\(thinkingPct) of output, by chars)
        Truncated:      \(truncated.isEmpty ? "none" : "⚠️ \(truncated.count) batch(es)")
        Parse failures: \(parseFailures.isEmpty ? "none" : "⚠️ \(parseFailures.count) batch(es)")
        ══════════════════════════════════════════════
        """)
    }

    static let internalPageMarker = "<<<PAGE_"
    static let internalPageMarkerSuffix = ">>>"

    static func stripRunningHeaders(_ text: String) -> String {
        let pages = text.components(separatedBy: PDFParser.pageBreakMarker)
        guard pages.count >= 4 else { return text }

        func normalized(_ line: String) -> String {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return "" }
            var out = ""
            var inDigits = false
            for ch in trimmed {
                if ch.isNumber {
                    if !inDigits { out.append("#"); inDigits = true }
                } else {
                    out.append(ch)
                    inDigits = false
                }
            }
            return out
        }

        var pageCount: [String: Int] = [:]
        for page in pages {
            var seen = Set<String>()
            for line in page.components(separatedBy: .newlines) {
                let key = normalized(line)
                if !key.isEmpty { seen.insert(key) }
            }
            for key in seen { pageCount[key, default: 0] += 1 }
        }

        let threshold = Int((Double(pages.count) * 0.6).rounded())
        let furniture = Set(pageCount.filter { $0.value >= threshold }.keys)
        guard !furniture.isEmpty else { return text }

        let cleanedPages = pages.map { page -> String in
            page.components(separatedBy: .newlines)
                .filter { !furniture.contains(normalized($0)) }
                .joined(separator: "\n")
        }

        print("🧽 [BOILERPLATE] Removed \(furniture.count) running header/footer line(s) present on ≥\(threshold)/\(pages.count) pages")
        return cleanedPages.joined(separator: PDFParser.pageBreakMarker)
    }

    private static func chunkByPages(_ text: String, pagesPerBatch: Int = 2) -> [PageBatch] {
        let pages = text.components(separatedBy: PDFParser.pageBreakMarker)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }

        guard pages.count > 1 else {
            let tagged = "[Page 1]:\n" + text
            return [PageBatch(text: tagged, pageRangeLabel: "Full Document", startPageNumber: 1, endPageNumber: 1)]
        }

        var batches: [PageBatch] = []
        var index = 0

        while index < pages.count {
            let end = min(index + pagesPerBatch, pages.count)
            let batchText = (index..<end)
                .filter { !pages[$0].isEmpty }
                .map { offset in
                    "\(internalPageMarker)\(offset + 1)\(internalPageMarkerSuffix)\n" + pages[offset]
                }
                .joined(separator: "\n\n")

            if !batchText.isEmpty {
                let rangeLabel = (end - index == 1)
                    ? "page \(index + 1)"
                    : "pages \(index + 1)–\(end)"
                batches.append(PageBatch(text: batchText, pageRangeLabel: rangeLabel, startPageNumber: index + 1, endPageNumber: end))
            }
            index = end
        }

        return batches
    }

    // MARK: - Page & Citation Resolvers

    private static func resolvePageNumbers(_ cases: [CitedCase], pageTexts: [String]) -> [CitedCase] {
        guard !pageTexts.isEmpty else { return cases }
        let lowered = pageTexts.map { $0.lowercased() }

        return cases.map { c in
            guard !c.isSelfReference else { return c }

            let tokens = nameTokens(c.caseName)
            guard !tokens.isEmpty else { return c }

            var occurrences: [Int] = []
            for (i, page) in lowered.enumerated() where tokens.allSatisfy({ page.contains($0) }) {
                occurrences.append(i + 1)
            }
            guard !occurrences.isEmpty else { return c }

            let hint = c.pageNumber ?? occurrences[0]
            let resolved = occurrences.min { abs($0 - hint) < abs($1 - hint) } ?? occurrences[0]
            guard resolved != c.pageNumber else { return c }

            print("📄 [PAGE FIX] \(c.caseName): model said \(c.pageNumber.map(String.init) ?? "nil") → text says \(resolved)")
            return CitedCase(
                caseName: c.caseName,
                citation: c.citation,
                year: c.year,
                court: c.court,
                context: c.context,
                pageNumber: resolved,
                isSelfReference: c.isSelfReference
            )
        }
    }

    /// Match citation and party names against PDF text to get exact verbatim string for highlighting
    /// Match citation against PDF text to get exact verbatim string for highlighting.
    /// Citation-name pollution avoid karne ke liye sirf citation tokens match kiye jaate hain — case name kabhi involve nahi hoti.
    private static func resolveExactPDFCitation(_ cases: [CitedCase], pageTexts: [String]) -> [CitedCase] {
        guard !pageTexts.isEmpty else { return cases }

        return cases.map { c in
            guard !c.isSelfReference,
                  let citation = c.citation?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !citation.isEmpty else {
                return c
            }

            let citationTokens = citation
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty }
            guard citationTokens.count >= 2 else { return c }

            let gap = "[\\s\\(\\)\\[\\]\\.,\\-–/]+"

            // 🔹 Build EXACT pattern (as before)
            let exactPattern = citationTokens
                .map { NSRegularExpression.escapedPattern(for: $0) }
                .joined(separator: gap)
                + "[\\)\\]\\.]{0,3}"

            // 🔹 Build FUZZY pattern: numeric tokens stay exact, alphabetic "reporter" tokens
            //    (short, all-letters, e.g. SCC/ELT/ITR/SSC) become a loose wildcard
            //    so OCR/typo mismatches like SSC vs SCC still match, since numbers anchor specificity.
            let fuzzyTokens = citationTokens.map { token -> String in
                let isNumeric = token.allSatisfy { $0.isNumber }
                let isShortAlpha = !isNumeric && token.count <= 6 && token.allSatisfy { $0.isLetter }
                if isShortAlpha {
                    return "[A-Za-z]{2,6}"   // reporter-code wildcard
                }
                return NSRegularExpression.escapedPattern(for: token)
            }
            let hasNumericAnchor = citationTokens.contains { $0.allSatisfy { $0.isNumber } }
            let fuzzyPattern = fuzzyTokens.joined(separator: gap) + "[\\)\\]\\.]{0,3}"

            guard let exactRegex = try? NSRegularExpression(pattern: exactPattern, options: [.caseInsensitive]) else {
                return c
            }
            // Only build fuzzy regex if there's at least one numeric anchor (avoid over-matching)
            let fuzzyRegex = hasNumericAnchor
                ? try? NSRegularExpression(pattern: fuzzyPattern, options: [.caseInsensitive])
                : nil

            let targetPage = (c.pageNumber ?? 1) - 1
            var searchPages: [Int] = []
            if targetPage >= 0 && targetPage < pageTexts.count { searchPages.append(targetPage) }
            if targetPage - 1 >= 0 { searchPages.append(targetPage - 1) }
            if targetPage + 1 < pageTexts.count { searchPages.append(targetPage + 1) }

            func clean(_ raw: String) -> String {
                var s = raw
                    .components(separatedBy: .whitespacesAndNewlines)
                    .filter { !$0.isEmpty }
                    .joined(separator: " ")
                if s.hasPrefix("[") && s.hasSuffix("]") {
                    s = String(s.dropFirst().dropLast()).trimmingCharacters(in: .whitespaces)
                }
                return s
            }

            // 1️⃣ Try EXACT match first
            for p in searchPages {
                let text = pageTexts[p]
                let ns = text as NSString
                if let match = exactRegex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) {
                    let rawMatchedText = ns.substring(with: match.range)
                    let displayText = clean(rawMatchedText)
                    print("🎯 [CITATION EXACT MATCH] Page \(p + 1): '\(citation)' -> '\(displayText)'")
                    return CitedCase(
                        caseName: c.caseName, citation: displayText, year: c.year, court: c.court,
                        context: c.context, pageNumber: p + 1, isSelfReference: c.isSelfReference,
                        pdfHighlightText: rawMatchedText
                    )
                }
            }

            // 2️⃣ Try FUZZY match (tolerates reporter-code OCR/typo mismatches)
            if let fuzzyRegex {
                for p in searchPages {
                    let text = pageTexts[p]
                    let ns = text as NSString
                    if let match = fuzzyRegex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) {
                        let rawMatchedText = ns.substring(with: match.range)
                        let displayText = clean(rawMatchedText)
                        print("🎯 [CITATION FUZZY MATCH] Page \(p + 1): '\(citation)' -> '\(displayText)'")
                        return CitedCase(
                            caseName: c.caseName, citation: displayText, year: c.year, court: c.court,
                            context: c.context, pageNumber: p + 1, isSelfReference: c.isSelfReference,
                            pdfHighlightText: rawMatchedText
                        )
                    }
                }
            }

            // 3️⃣ Final fallback
            let fallbackClean = clean(citation)
            print("⚠️ [NO PDF MATCH] '\(citation)' not found on pages \(searchPages.map { $0 + 1 })")
            return CitedCase(
                caseName: c.caseName, citation: fallbackClean, year: c.year, court: c.court,
                context: c.context, pageNumber: c.pageNumber, isSelfReference: c.isSelfReference,
                pdfHighlightText: fallbackClean
            )
        }
    }
             
    private static func extractFootnotes(from pageTexts: [String]) -> [Int: [String: String]] {
        var pageFootnotes: [Int: [String: String]] = [:]

        let pattern = "(?m)^\\s*(\\d{1,2})[\\.\\s\\t]+([\\(\\[]?(?:19|20)\\d{2}.*?(?:SCC|AIR|SCR|ITR|ELT|SCALE|JT|INSC|Supp|Online|OnLine).*)$"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
            return pageFootnotes
        }

        for (idx, pageText) in pageTexts.enumerated() {
            let pageNum = idx + 1
            var footnotes: [String: String] = [:]
            let ns = pageText as NSString
            let matches = regex.matches(in: pageText, range: NSRange(location: 0, length: ns.length))

            for m in matches where m.numberOfRanges >= 3 {
                let numStr = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                let citationStr = ns.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                if !numStr.isEmpty && !citationStr.isEmpty {
                    footnotes[numStr] = citationStr
                }
            }

            if footnotes.isEmpty {
                let fallbackPattern = "(?m)^\\s*(\\d{1,2})[\\.\\s\\t]+((?:(?:\\(?\\d{4}\\)?)|(?:AIR|SCC|SCR|ITR|ELT|SCALE|JT|INSC)).{5,100})$"
                if let fallbackRegex = try? NSRegularExpression(pattern: fallbackPattern, options: [.caseInsensitive]) {
                    let fallbackMatches = fallbackRegex.matches(in: pageText, range: NSRange(location: 0, length: ns.length))
                    for m in fallbackMatches where m.numberOfRanges >= 3 {
                        let numStr = ns.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                        let citationStr = ns.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !numStr.isEmpty && !citationStr.isEmpty {
                            footnotes[numStr] = citationStr
                        }
                    }
                }
            }

            if !footnotes.isEmpty {
                pageFootnotes[pageNum] = footnotes
            }
        }
        return pageFootnotes
    }

    private static func resolveFootnoteCitations(_ cases: [CitedCase], pageTexts: [String]) -> [CitedCase] {
        guard !cases.isEmpty, !pageTexts.isEmpty else { return cases }
        let footnotesByPage = extractFootnotes(from: pageTexts)

        var allFootnotes: [String: String] = [:]
        for (_, map) in footnotesByPage {
            for (k, v) in map {
                allFootnotes[k] = v
            }
        }

        return cases.map { c in
            guard !c.isSelfReference else { return c }

            var footnoteKey: String? = nil
            if let citation = c.citation?.trimmingCharacters(in: .whitespacesAndNewlines) {
                let cleaned = citation.trimmingCharacters(in: CharacterSet(charactersIn: "()[]{}."))
                if let num = Int(cleaned), num > 0 && num < 100 {
                    footnoteKey = String(num)
                }
            }

            var cleanCaseName = c.caseName
            if footnoteKey == nil {
                let trailingDigitRegex = try? NSRegularExpression(pattern: "\\s+(\\d{1,2})$")
                let nsName = c.caseName as NSString
                if let match = trailingDigitRegex?.firstMatch(in: c.caseName, range: NSRange(location: 0, length: nsName.length)),
                   match.numberOfRanges >= 2 {
                    let digitStr = nsName.substring(with: match.range(at: 1))
                    footnoteKey = digitStr
                    cleanCaseName = nsName.substring(with: NSRange(location: 0, length: match.range.location)).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }

            guard let key = footnoteKey else { return c }

            let page = c.pageNumber ?? 1
            let resolvedCitation = footnotesByPage[page]?[key]
                ?? footnotesByPage[page - 1]?[key]
                ?? footnotesByPage[page + 1]?[key]
                ?? allFootnotes[key]

            guard let finalCitation = resolvedCitation else {
                return CitedCase(
                    caseName: cleanCaseName,
                    citation: c.citation,
                    year: c.year,
                    court: c.court,
                    context: c.context,
                    pageNumber: c.pageNumber,
                    isSelfReference: c.isSelfReference
                )
            }

            var finalYear = c.year
            if finalYear == nil || (finalYear ?? 0) < 1800 {
                let yearRegex = try? NSRegularExpression(pattern: "\\b(19\\d{2}|20\\d{2})\\b")
                let nsCit = finalCitation as NSString
                if let match = yearRegex?.firstMatch(in: finalCitation, range: NSRange(location: 0, length: nsCit.length)) {
                    finalYear = Int(nsCit.substring(with: match.range))
                }
            }

            print("📚 [FOOTNOTE RESOLVED] \(cleanCaseName): '\(c.citation ?? "nil")' -> '\(finalCitation)' (Year: \(finalYear.map(String.init) ?? "nil"))")

            return CitedCase(
                caseName: cleanCaseName,
                citation: finalCitation,
                year: finalYear,
                court: c.court ?? "Supreme Court of India",
                context: c.context,
                pageNumber: c.pageNumber,
                isSelfReference: c.isSelfReference
            )
        }
    }

    private static func removingPlaceholders(_ cases: [CitedCase]) -> [CitedCase] {
        cases.filter { c in
            let cleanName = c.caseName.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleanName.isEmpty else { return false }
            let lowered = cleanName.lowercased()
            return lowered != "none" && lowered != "unknown case"
        }
    }

    private static func nameTokens(_ name: String) -> Set<String> {
        let noise: Set<String> = ["versus", "state", "union", "india", "ltd", "limited",
                                  "anr", "another", "ors", "others", "the", "and", "bank",
                                  "company", "corporation", "pvt", "private",
                                  "petitioner", "petitioners", "respondent", "respondents",
                                  "appellant", "appellants", "applicant", "applicants",
                                  "plaintiff", "plaintiffs", "defendant", "defendants",
                                  "through", "secretary", "govt", "government", "with"]
        let words = name.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 3 && !noise.contains($0) }
        return Set(words)
    }

    private static func cleanedIfPossible(_ output: String) -> String {
        guard let parsed = DocumentExtractionResult.parse(from: output) else { return output }
        let result = DocumentExtractionResult(citedCases: removingPlaceholders(parsed.citedCases))
        guard let data = try? JSONEncoder().encode(result),
              let json = String(data: data, encoding: .utf8) else { return output }
        return json
    }

    private static func mergeChunkOutputs(_ outputs: [String]) -> String {
        var allCases: [CitedCase] = []

        for output in outputs {
            if let parsed = DocumentExtractionResult.parse(from: output) {
                allCases.append(contentsOf: parsed.citedCases)
            }
        }

        let cleaned = removingPlaceholders(allCases)

        if cleaned.isEmpty {
            return "{\"citedCases\":[]}"
        }

        let result = DocumentExtractionResult(citedCases: cleaned)
        if let data = try? JSONEncoder().encode(result), let json = String(data: data, encoding: .utf8) {
            return json
        }
        return "{\"citedCases\":[]}"
    }

    private func generateForChunk(
        container: ModelContainer,
        activePrompt: String,
        chunkText: String,
        chunkLabel: String?,
        thinkingEnabled: Bool,
        onThinkingToken: @MainActor @escaping (String) -> Void,
        onToken: @MainActor @escaping (String) -> Void
    ) async throws -> ChunkResult {

        let userContent = chunkLabel != nil
            ? "Extract the file\n\n\(chunkLabel!)\n\n\(chunkText)"
            : "Extract the file\n\n\(chunkText)"

        return try await container.perform { context in
            let chatMessages: [Chat.Message] = [
                Chat.Message(role: .system, content: activePrompt),
                Chat.Message(role: .user, content: userContent)
            ]
            
            let input = try await context.processor.prepare(
                input: UserInput(
                    prompt: .chat(chatMessages),
                    additionalContext: [
                        "enable_thinking": thinkingEnabled
                    ]
                )
            )

            var generateParameters = GenerateParameters(temperature: 0.0)
            generateParameters.maxTokens = 2048

            generateParameters.repetitionPenalty = nil
            generateParameters.presencePenalty = nil
            generateParameters.frequencyPenalty = nil

            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: generateParameters,
                context: context
            )

            var fullText = ""
            var thinkingText = ""
            var metrics = await ChunkMetrics()
            
            let parser = await StreamingReasoningParser(
                thinkingEnabled: thinkingEnabled,
                onThinking: { t in
                    thinkingText += t
                    Task { @MainActor in
                        onThinkingToken(t)
                    }
                },
                onNormal: { t in
                    fullText += t
                    Task { @MainActor in
                        onToken(t)
                    }
                }
            )

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    await parser.process(text)

                case .info(let info):
                    metrics.promptTokens = info.promptTokenCount
                    metrics.generatedTokens = info.generationTokenCount
                    metrics.prefillSeconds = info.promptTime
                    metrics.decodeSeconds = info.generateTime
                    switch info.stopReason {
                    case .stop:
                        metrics.stopReasonText = "stop"
                    case .length:
                        metrics.stopReasonText = "length"
                        metrics.hitTokenLimit = true
                    case .cancelled:
                        metrics.stopReasonText = "cancelled"
                    }

                case .toolCall:
                    break
                }
            }
            
            await parser.finalize()

            metrics.thinkingChars = thinkingText.count
            metrics.responseChars = fullText.count

            let preview = String(fullText.prefix(300))
            print("🔍 [RAW OUTPUT] \(fullText.count) chars — preview: \(preview)")

            return ChunkResult(text: fullText, metrics: metrics, thinkingText: thinkingText)
        }
    }

    private nonisolated static let gemmaChannelReasoningConfig = ReasoningConfig(
        startDelimiter: "<|channel>thought",
        endDelimiter: "<channel|>",
        promptStrategy: .alwaysOn
    )

    enum LLMError: LocalizedError {
        case noRepoConfigured
        var errorDescription: String? {
            "No Hugging Face repo ID is set for this model. Add one in LLMModel.hubRepoID."
        }
    }
}
