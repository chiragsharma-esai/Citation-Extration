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
    var extractedOutput: String
    var outputFormat: OutputFormat
    var generationTimeText: String
    var thinkingText: String = ""
    var thinkingEnabled: Bool = true
    var model: LLMModel
}

// MARK: - Structured Extraction Schema
struct CitedCase: Codable, Identifiable, Hashable {
    var id: String { caseName + (citation ?? "") + (year.map(String.init) ?? "") }
    let caseName: String
    let citation: String?
    let year: Int?
    let court: String?
    let context: String?

    private enum CodingKeys: String, CodingKey {
        case caseName, citation, year, court, context
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        caseName = (try? container.decode(String.self, forKey: .caseName)) ?? "Unknown Case"
        citation = Self.decodeLenientString(container, .citation)
        court = Self.decodeLenientString(container, .court)
        context = Self.decodeLenientString(container, .context)
        year = Self.decodeLenientInt(container, .year)
    }

    init(caseName: String, citation: String?, year: Int?, court: String?, context: String?) {
        self.caseName = caseName
        self.citation = citation
        self.year = year
        self.court = court
        self.context = context
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

        guard let firstBrace = text.firstIndex(of: "{"),
              let lastBrace = text.lastIndex(of: "}") else { return nil }
        text = String(text[firstBrace...lastBrace])

        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DocumentExtractionResult.self, from: data)
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

    var hubRepoID: String?{
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
    private var upstream: any Tokenizers.Tokenizer {
        didSet {
            // Unused but required
        }
    }

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

enum OutputFormat: String, CaseIterable, Identifiable {
    case json = "JSON"
    case text = "Text"
    var id: String { rawValue }
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

// MARK: - STEP 1: UNIVERSAL HIGH-PRECISION CITATION SCANNER
enum CitationScanner {
    
    static let defaultCharsBefore = 200
    static let defaultCharsAfter = 200

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
        "SarPCJ", "ACE", "WLR", "CalLT", "MWN", "TAC",
        "EWHC", "UKSC"
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
        //"SLP(C)", "SLP(Crl)", "SLP", "W.P.(C)", "WP(C)", "W.P.(Crl)", "WP(Crl)", "W.P.", "WP",
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

        // Standard reporters & neutral citations
        let standardTokens = (reporterTokens + neutralTokens)
            .sorted { ($0.count, $0) > ($1.count, $1) }
            .map(tolerantAcronym)
            .joined(separator: "|")

        // Docket patterns e.g. SLP(C) No. 6092 of 2025, CS(COMM) 583/2025, W.P.(C) 1206/2025
        let docketEscaped = docketTokens
            .sorted { ($0.count, $0) > ($1.count, $1) }
            .map { NSRegularExpression.escapedPattern(for: $0) }
            .joined(separator: "|")
        let docketPattern = "(?i:(?:\(docketEscaped)))\\s*(?:\\(?[A-Za-z]+\\)?\\s*)?(?:No\\.?|Nos\\.?)?\\s*\\d+"

        // Dedicated pattern for SCC OnLine variants: "SCC OnLineBom 1262", "SCC OnLine Del 4029", "SCC Online 2024"
        let sccOnlinePattern = "(?i:SCC\\s*OnLine(?:\\s*[A-Za-z]{2,6})?\\s*\\d+)"

        var combinedPatterns: [String] = [
            "(?<![A-Za-z])(?:\(standardTokens))(?:\\s*\\([A-Za-z&.\\- ]{1,14}\\))?(?![A-Za-z])",
            sccOnlinePattern,
            docketPattern
        ]

        if includePartyMarker {
            // Party markers: v., vs., versus, In Re:, titled '...'
            let partyPattern = "(?<=\\w[\\s\\-,])(?i:v|vs|v/s)\\.?(?=[\\s\\-,;:]|$)|(?i:\\bversus\\b)|(?i:\\bIn\\s+Re:?\\b)|(?i:\\btitled\\s+['\"‘][A-Za-z])"
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
        
        // Snap start backwards to nearest whitespace or newline (up to 40 chars search)
        if start > 0 {
            var searchIdx = start
            while searchIdx > 0 && searchIdx > start - 40 {
                let char = text.character(at: searchIdx)
                if char == 10 || char == 32 { // \n or space
                    start = searchIdx + 1
                    break
                }
                searchIdx -= 1
            }
        }
        
        // Snap end forwards to nearest whitespace or newline (up to 40 chars search)
        if end < text.length {
            var searchIdx = end
            while searchIdx < text.length && searchIdx < end + 40 {
                let char = text.character(at: searchIdx)
                if char == 10 || char == 32 { // \n or space
                    end = searchIdx
                    break
                }
                searchIdx += 1
            }
        }
        
        return NSRange(location: start, length: max(0, end - start))
    }

    static func candidatePassages(
        in text: String,
        charsBefore: Int = defaultCharsBefore,
        charsAfter: Int = defaultCharsAfter
    ) -> String? {
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        let matches = citationRegex.matches(in: text, range: full)
        guard !matches.isEmpty else { return nil }

        // 1. Expand character windows around each regex hit
        var rawRanges: [NSRange] = []
        for match in matches {
            let start = max(0, match.range.location - charsBefore)
            let end = min(ns.length, match.range.location + match.range.length + charsAfter)
            rawRanges.append(NSRange(location: start, length: end - start))
        }

        // 2. Merge overlapping / nearby character windows (within 60 characters)
        let sortedRanges = rawRanges.sorted { $0.location < $1.location }
        var mergedRanges: [NSRange] = []
        for range in sortedRanges {
            if let last = mergedRanges.last {
                let lastEnd = last.location + last.length
                if range.location <= lastEnd + 60 {
                    mergedRanges[mergedRanges.count - 1].length = max(lastEnd, range.location + range.length) - last.location
                } else {
                    mergedRanges.append(range)
                }
            } else {
                mergedRanges.append(range)
            }
        }

        // 3. Snap to clean word boundaries and extract candidate text passages
        var passages = mergedRanges.compactMap { range -> String? in
            let snapped = snapToWordBoundaries(in: ns, range: range)
            guard snapped.length > 0 else { return nil }
            let passage = ns.substring(with: snapped).trimmingCharacters(in: .whitespacesAndNewlines)
            if isPureHeaderPassage(passage) {
                return nil
            }
            return passage.isEmpty ? nil : passage
        }

        guard !passages.isEmpty else { return nil }

        // 4. Preserve Footnote block at the end of the batch
        let lines = text.components(separatedBy: .newlines)
        let footnoteLines = lines.filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let firstChar = trimmed.first, firstChar.isNumber else { return false }
            return trimmed.contains("SCC") || trimmed.contains("AIR") || trimmed.contains("INSC") || trimmed.contains("Supp") || trimmed.contains("Judges") || trimmed.contains("EWHC")
        }

        if !footnoteLines.isEmpty {
            let footnotesBlock = "FOOTNOTES FOR THIS PAGE:\n" + footnoteLines.joined(separator: "\n")
            passages.append(footnotesBlock)
        }

        return passages.joined(separator: "\n\n[…]\n\n")
    }
}


// MARK: - HIGH-PRECISION STREAM ROUTER
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
                } else {
                    let maxKeep = 100
                    if buffer.count > maxKeep {
                        buffer = String(buffer.suffix(maxKeep))
                    }
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

    let textOnlySystemPrompt = """
        You are an expert Legal AI Assistant specializing in Indian jurisprudence. Your task is to extract ALL cited legal precedents, case laws, and referenced court proceedings from the provided document text with 100% accuracy.

        CRITICAL EXTRACTION RULES (APPLIES TO ALL INDIAN LEGAL DOCUMENTS):

            1. FIELD DEFINITIONS & MAPPING:
               - [Case Name]: The title of the parties in adversarial or title format (e.g., "Party A v. Party B", "In Re: XYZ Ltd.", "State of X v. Person Y").
               - [Citation / Case Number]: Extract either standard volume reporters (e.g., SCC, AIR, ITR, ELT, SCR, SCC OnLine, Neutral Citations like YYYY:DHC:XXXX) OR court docket/filing numbers (e.g., "SLP(C) No. XXXX/YYYY", "W.P.(C) XXXX/YYYY", "CS(COMM) XXX/YYYY", "Crl.A. No. XXX/YYYY"). If no citation or docket number is mentioned in the text, write 'N/A'.
               - [Context/Reason]: A concise 1-sentence summary of the legal principle, proposition, or reason the case was cited.

            2. SEPARATION OF DOCKET NUMBERS AND CASE TITLES:
               - When a proceeding is introduced by its filing/docket number followed by "titled" or "in the case of" (e.g., "...in W.P.(C) 1234/2021 titled 'ABC Corp v. Union of India'..."):
                 * Put the party names ("ABC Corp v. Union of India") in [Case Name].
                 * Put the filing number ("W.P.(C) 1234/2021") in [Citation or Case Number].
                 * NEVER mark the citation as 'N/A' if a valid court filing/docket number is present.

            3. DISTINCT PROCEEDINGS FOR SAME PARTIES:
               - If the SAME party names appear with DIFFERENT citations or case numbers (such as an earlier Trial Court suit, a High Court Writ Petition, and a Supreme Court SLP, or multiple separate orders), extract EACH proceeding as a SEPARATE entry.
               - Only deduplicate when BOTH the Case Name AND the Citation/Docket number refer to the exact same proceeding.

            4. FLEXIBLE SYNTAX & REVERSE PHRASING:
               - Recognize cases where the citation appears before the case name (e.g., "In (2020) 2 SCC 100, Party A v. Party B..." or "The decision at citation (2015) 3 ITR 50 is namely ABC Ltd. v. CIT"). Extract the case name and citation into their respective fields regardless of word order.

            5. FULL ENTITY NAMES ACROSS LINE BREAKS:
               - Always capture the complete legal entity name. Do not truncate names if they continue onto subsequent lines or across punctuation (e.g., include full company suffixes like "Pvt. Ltd.", "LLP", or full party descriptions).

            6. EXCLUSIONS (DO NOT EXTRACT):
               - EXCLUDE the primary heading/caption and case number of the main document currently being read.
               - EXCLUDE running page footers, pagination lines (e.g., "Page X of Y"), digital signature blocks, and statutory acts/sections.

        OUTPUT FORMAT:
        [Case Name] — [Citation or Case Number] — [Context/Reason for citation]

        (If no cited cases are found in the text, output: No cited cases found.)

        OUTPUT:
        """
    
    let jsonSystemPrompt = """
    You are an expert Legal AI Assistant specializing in Indian jurisprudence and legal document analysis.
    Your task is to carefully read through Indian court filing documents and extract every legal case cited as a precedent or reference.

    Follow these strict guidelines:
    1. Identify Case Names: Look for standard adversarial formats (e.g., "X v. Y", "X vs. Y", "In Re: X").
    2. Identify Citations: Look for standard Indian legal reporters (SCC, AIR, SCR, SCALE, JT, SCC OnLine, Neutral citations).
    3. Context: Briefly summarize the legal principle or reason why the case was cited.
    4. Missing Data: If a detail is missing, return null. Do not hallucinate.
    5. No Repeats: Emit each unique case only once.
    6. NEVER extract footnote numbers or naked years as case names. Combine them into the citation field.

    OUTPUT FORMAT:
    Respond with ONLY a single valid JSON object, and nothing else (no ``` markdown fences).

    type CitedCase = {
      caseName: string;
      citation: string | null;
      year: number | null;
      court: string | null;
      context: string | null;
    };
    type DocumentExtractionResult = {
      citedCases: CitedCase[];
    };
    """

    let passageExcerptNote = """

    INPUT FORMAT NOTE:
    The text below consists of candidate excerpts surrounding cited cases joined with "[…]". Extract the cases cited directly from these excerpts.
    """

    let strictThinkingDirective = """

    REASONING GUIDELINE (Inside <|channel>thought):
    - Write your thoughts as a natural, conversational stream-of-consciousness (thinking out loud).
    - Avoid robotic templates, repeating lines, "1. Analyze", "Step 1", or structured tables.
    - Start directly with conversational phrases like: "Let's look at paragraph...", "Scanning this chunk...", "I see footnote 1 maps to..."
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
        outputFormat: OutputFormat,
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
        let basePrompt = outputFormat == .json ? jsonSystemPrompt : textOnlySystemPrompt
        
        let promptWithThinking = thinkingEnabled ? (basePrompt + strictThinkingDirective) : basePrompt
        let activePrompt = preFilterEnabled ? (promptWithThinking + passageExcerptNote) : promptWithThinking

        let totalStartTime = CFAbsoluteTimeGetCurrent()

        let batches = Self.chunkByPages(pdfText)
        let isSingleBatch = batches.count == 1

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
                textToSend = passages
            }

            let batchSource = preFilterEnabled
                ? "excerpts taken from pages \(batch.pageRangeLabel)"
                : "a batch of pages (\(batch.pageRangeLabel))"
            let label: String? = isSingleBatch
                ? nil
                : "This is \(batchSource) of a larger legal document, part \(index + 1) of \(batches.count) overall. Extract cited cases found in THIS BATCH only."

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

            if outputFormat == .json {
                if let parsed = DocumentExtractionResult.parse(from: result.text) {
                    metrics.parseStatus = .parsed(count: parsed.citedCases.count)
                } else {
                    metrics.parseStatus = .failed
                }
            }

            chunkOutputs.append(result.text)
            chunkThinking.append(result.thinkingText)
            allMetrics.append(metrics)

            print("✅ [COMPLETE] \(metrics.summaryLine)")
        }

        let finalResult: String
        if chunkOutputs.isEmpty {
            finalResult = outputFormat == .json
                ? "{\"citedCases\":[]}"
                : "No cited cases found in this document."
        } else if isSingleBatch {
            finalResult = Self.dedupedIfPossible(chunkOutputs[0], outputFormat: outputFormat)
        } else {
            statusText = "Merging results from \(chunkOutputs.count) page-batches..."
            finalResult = Self.mergeChunkOutputs(chunkOutputs, outputFormat: outputFormat)
        }

        let finalThinking = chunkThinking.joined(separator: "\n\n")

        let totalTimeTaken = CFAbsoluteTimeGetCurrent() - totalStartTime
        Self.printRunSummary(allMetrics, totalSeconds: totalTimeTaken, outputFormat: outputFormat)

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
    }

    private static func printRunSummary(_ metrics: [ChunkMetrics], totalSeconds: Double, outputFormat: OutputFormat) {
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

        ═══════════ RUN SUMMARY (\(outputFormat.rawValue)) ═══════════
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

        for m in truncated {
            print("⚠️  TRUNCATED: batch \(m.label) was cut off by maxTokens after \(m.generatedTokens) generated tokens (~\(m.estimatedThinkingTokens) of them thinking) — citations near the end of this batch were probably lost.")
        }
        for m in parseFailures {
            print("⚠️  PARSE FAILED: batch \(m.label) contributed 0 cases to the merged result.")
        }
    }

    private static func chunkByPages(_ text: String, pagesPerBatch: Int = 2) -> [PageBatch] {
        let pages = text.components(separatedBy: PDFParser.pageBreakMarker)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard pages.count > 1 else {
            return [PageBatch(text: text, pageRangeLabel: "Full Document")]
        }

        var batches: [PageBatch] = []
        var index = 0
        
        while index < pages.count {
            let end = min(index + pagesPerBatch, pages.count)
            let batchPages = pages[index..<end]
            let batchText = batchPages.joined(separator: "\n\n")
            
            let rangeLabel = (end - index == 1)
                ? "page \(index + 1)"
                : "pages \(index + 1)–\(end)"

            batches.append(PageBatch(text: batchText, pageRangeLabel: rangeLabel))
            index = end
        }
        
        return batches
    }

    private static func dedupeKey(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    private static func dedupeCitedCases(_ cases: [CitedCase]) -> [CitedCase] {
            var uniqueCases: [CitedCase] = []
            var seenCompositeKeys = Set<String>()

            for c in cases {
                let cleanName = c.caseName.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !cleanName.isEmpty,
                      cleanName.lowercased() != "none", 
                      cleanName.lowercased() != "unknown case" else { continue }

                let nameKey = dedupeKey(cleanName)
                let citationKey = dedupeKey(c.citation ?? "nocitation")
                
                // FIX: Composite key checks BOTH Name AND Citation/Docket Number
                // This ensures W.P.(C) 1206/2025 and SLP(C) 8544/2025 are BOTH preserved!
                let compositeKey = nameKey + "_" + citationKey

                if !seenCompositeKeys.contains(compositeKey) {
                    seenCompositeKeys.insert(compositeKey)
                    uniqueCases.append(c)
                }
            }
            return uniqueCases
        }

    private static func nameTokens(_ name: String) -> Set<String> {
        let noise: Set<String> = ["versus", "state", "union", "india", "ltd", "limited",
                                  "anr", "another", "ors", "others", "the", "and", "bank",
                                  "company", "corporation", "pvt", "private"]
        let words = name.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 3 && !noise.contains($0) }
        return Set(words)
    }

    private static func dedupedIfPossible(_ output: String, outputFormat: OutputFormat) -> String {
        guard outputFormat == .json,
              let parsed = DocumentExtractionResult.parse(from: output) else { return output }
        let result = DocumentExtractionResult(citedCases: dedupeCitedCases(parsed.citedCases))
        guard let data = try? JSONEncoder().encode(result),
              let json = String(data: data, encoding: .utf8) else { return output }
        return json
    }

    private static func mergeChunkOutputs(_ outputs: [String], outputFormat: OutputFormat) -> String {
        var allCases: [CitedCase] = []

        for output in outputs {
            if let parsed = DocumentExtractionResult.parse(from: output) {
                allCases.append(contentsOf: parsed.citedCases)
            } else {
                let lines = output.components(separatedBy: .newlines)
                for line in lines {
                    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty,
                          !trimmed.hasPrefix("---"),
                          !trimmed.lowercased().contains("no legal cases were cited"),
                          !trimmed.lowercased().contains("no cited cases"),
                          trimmed.uppercased() != "NONE" else { continue }
                    
                    var normalizedLine = trimmed
                    if normalizedLine.hasPrefix("Case Name —") || normalizedLine.hasPrefix("Case Name -") {
                        normalizedLine = normalizedLine.replacingOccurrences(of: "Case Name —", with: "")
                                                       .replacingOccurrences(of: "Case Name -", with: "")
                                                       .trimmingCharacters(in: .whitespacesAndNewlines)
                    }

                    guard normalizedLine.contains("—") || normalizedLine.contains("-") || normalizedLine.contains("SCC") || normalizedLine.contains("AIR") else { continue }

                    let parts = normalizedLine.components(separatedBy: "—")
                    let name = parts.first?.trimmingCharacters(in: .whitespacesAndNewlines) ?? normalizedLine
                    let cit = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespacesAndNewlines) : nil
                    let ctx = parts.count > 2 ? parts[2].trimmingCharacters(in: .whitespacesAndNewlines) : nil
                    
                    let normalizedName = name.replacingOccurrences(of: "Case Name —", with: "")
                                             .replacingOccurrences(of: "Case Name -", with: "")
                                             .trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    allCases.append(CitedCase(caseName: normalizedName, citation: cit, year: nil, court: nil, context: ctx))
                }
            }
        }

        let deduped = dedupeCitedCases(allCases)

        if deduped.isEmpty {
            return outputFormat == .json ? "{\"citedCases\":[]}" : "No cited cases found in this document."
        }

        switch outputFormat {
        case .json:
            let result = DocumentExtractionResult(citedCases: deduped)
            if let data = try? JSONEncoder().encode(result), let json = String(data: data, encoding: .utf8) {
                return json
            }
            return "{\"citedCases\":[]}"

        case .text:
            return deduped.map { c in
                var line = "Case Name — " + c.caseName
                if let citation = c.citation, !citation.isEmpty { line += " — \(citation)" }
                if let context = c.context, !context.isEmpty { line += " — \(context)" }
                return line
            }.joined(separator: "\n")
        }
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
                        "enable_thinking": thinkingEnabled,
                        //"thinking_budget": 192,
                      //    "max_thinking_tokens": 192
                    ]
                )
            )

            var generateParameters = GenerateParameters(temperature: 0.0)
            generateParameters.topP = 0.95
            generateParameters.repetitionPenalty = 1.0
            generateParameters.maxTokens = 2048

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
