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

// MARK: - Data Models
struct ChatHistory: Identifiable, Hashable {
    let id = UUID()
    var title: String
    var pdfText: String
    var extractedOutput: String
    var outputFormat: OutputFormat
    var generationTimeText: String
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

    static func extractText(from url: URL) -> String? {
        guard let pdf = PDFDocument(url: url) else { return nil }
        var pageTexts: [String] = []
        for i in 0..<pdf.pageCount {
            guard let page = pdf.page(at: i) else { continue }
            pageTexts.append(page.string ?? "")
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
    private let upstream: any Tokenizers.Tokenizer

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
}

// MARK: - 🚀 STEP 1: UNCOMMENTED & OPTIMIZED CITATION SCANNER
// Correctly skips batches with only header/procedural information (e.g. Batch 1)
/*struct CitationScanner {
    private static let citationRegex: NSRegularExpression? = {
        let pattern = #"""
        (?ix)
        \b(
            # Adversarial patterns (Party v. Party)
            \b[A-Z][A-Za-z0-9\.\s]{2,35}\s+(?:v\.|vs\.|versus)\s+[A-Z][A-Za-z0-9\.\s]{2,35}\b
            |
            # Standard Indian & Commonwealth Reporters
            (?:\(\d{4}\)|\b\d{4}\b)\s*(\d+)?\s*(?:SCC|AIR|SCR|SCALE|JT|DLT|Bom\s*CR|Bom\s*LR|KLT|MLJ|All\s*LJ|BLJ|ILR|Cr\.?L\.?J|EWHC|UKSC)\s*(?:\([A-Z]+\))?\s*\d+
            |
            # SCC OnLine & Neutral Citations
            \b\d{4}\s*(?:SCC\s+OnLine|INSC|\/[A-Z]{3,4}\/|EWHC)\s+[A-Z0-9\s]+\b
        )
        """#
        return try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }()

    static func candidatePassages(in text: String, windowSize: Int = 400) -> String? {
        guard let regex = citationRegex else { return text }
        let nsString = text as NSString
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))

        guard !matches.isEmpty else { return nil }

        var mergedRanges: [NSRange] = []

        for match in matches {
            let start = max(0, match.range.location - windowSize)
            let end = min(nsString.length, match.range.location + match.range.length + windowSize)
            let expandedRange = NSRange(location: start, length: end - start)

            if let last = mergedRanges.last, NSIntersectionRange(last, expandedRange).length > 0 || last.location + last.length >= expandedRange.location {
                let unionRange = NSUnionRange(last, expandedRange)
                mergedRanges[mergedRanges.count - 1] = unionRange
            } else {
                mergedRanges.append(expandedRange)
            }
        }

        let passages = mergedRanges.map { nsString.substring(with: $0).trimmingCharacters(in: .whitespacesAndNewlines) }
        return passages.joined(separator: "\n\n[…]\n\n")
    }
}*/

// MARK: - LLM Manager
@MainActor
class LLMManager: ObservableObject {
    @Published var isGenerating = false
    @Published var downloadProgress: Double = 0.0
    @Published var statusText: String = ""
    @Published var generationTimeText: String = ""

    let textOnlySystemPrompt = """
    You are an expert Legal AI Assistant specializing in Indian jurisprudence.
    Your ONLY task is to extract cited legal precedents and cases from the provided document excerpts.

    STRICT OUTPUT RULES:
    1. For each cited case, output strictly ONE line in this format:
       Case Name — Citation — Reason / Context
    2. Do NOT summarize the document facts or write background essays.
    3. DO NOT repeat cases.
    4. If an excerpt contains NO cited precedents, output strictly:
       NONE
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

    // 🚀 STEP 2: TIGHTER REASONING DIRECTIVE TO REDUCE 70%+ THINKING OVERHEAD
    let strictThinkingDirective = """

    CRITICAL REASONING INSTRUCTIONS (Inside Thinking Block):
    - Limit internal thinking to under 30 words total.
    - DO NOT quote text passages in your thinking.
    - Extract case name and citation directly, then immediately write the JSON/Output.
    """
    
    private var loadedContainers: [String: ModelContainer] = [:]

    private func loadContainer(repoID: String, architectureType: ModelArchitectureType) async throws -> ModelContainer {
        if let cached = loadedContainers[repoID] {
            return cached
        }

        statusText = "Downloading/Loading \(repoID)..."
        let configuration = ModelConfiguration(id: repoID)

        let progressClosure: @Sendable (Progress) -> Void = { progress in
            Task { @MainActor in
                self.downloadProgress = progress.fractionCompleted
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
        statusText = ""
        downloadProgress = 1.0
        return container
    }

    private static func warmup(container: ModelContainer) async {
        _ = try? await container.perform { context in
            let input = try await context.processor.prepare(
                input: UserInput(prompt: .text("1"))
            )
            var params = GenerateParameters(temperature: 0.0)
            params.maxTokens = 1

            let stream = try MLXLMCommon.generate(input: input, parameters: params, context: context)
            for await _ in stream {
                break
            }
        }
    }

    func generateStructuredOutput(pdfText: String, model: LLMModel, outputFormat: OutputFormat, thinkingEnabled: Bool = true, preFilterEnabled: Bool = true) async throws -> String {
        guard let repoID = model.hubRepoID else {
            throw LLMError.noRepoConfigured
        }

        isGenerating = true
        generationTimeText = "Processing..."
        defer { isGenerating = false }

        let container = try await loadContainer(repoID: repoID, architectureType: model.architectureType)
        let basePrompt = outputFormat == .json ? jsonSystemPrompt : textOnlySystemPrompt
        
        let promptWithThinking = thinkingEnabled ? (basePrompt + strictThinkingDirective) : basePrompt
        let activePrompt = preFilterEnabled ? (promptWithThinking + passageExcerptNote) : promptWithThinking

        let totalStartTime = CFAbsoluteTimeGetCurrent()

        let batches = Self.chunkByPages(pdfText)
        let isSingleBatch = batches.count == 1

        var chunkOutputs: [String] = []
        var allMetrics: [ChunkMetrics] = []

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
                container: container, activePrompt: activePrompt, chunkText: textToSend,
                chunkLabel: label, thinkingEnabled: thinkingEnabled
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

        return finalResult
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

    private static func chunkByPages(_ text: String, pagesPerBatch: Int = 5) -> [PageBatch] {
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
        var seenNames = Set<String>()
        var citationOwners: [String: Set<String>] = [:]
        var out: [CitedCase] = []

        for citedCase in cases {
            let nameKey = dedupeKey(citedCase.caseName)
            guard !nameKey.isEmpty else { continue }
            let citationKey = citedCase.citation.map(dedupeKey) ?? ""
            let nameTokens = Self.nameTokens(citedCase.caseName)

            var isDuplicate = seenNames.contains(nameKey)

            if !isDuplicate, !citationKey.isEmpty, let owners = citationOwners[citationKey] {
                isDuplicate = owners.isEmpty || !owners.isDisjoint(with: nameTokens)
            }

            seenNames.insert(nameKey)
            if !citationKey.isEmpty {
                citationOwners[citationKey, default: []].formUnion(nameTokens)
            }

            if isDuplicate { continue }
            out.append(citedCase)
        }
        return out
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
        switch outputFormat {
        case .json:
            var all: [CitedCase] = []
            for output in outputs {
                guard let parsed = DocumentExtractionResult.parse(from: output) else { continue }
                all.append(contentsOf: parsed.citedCases)
            }
            let result = DocumentExtractionResult(citedCases: dedupeCitedCases(all))
            guard let data = try? JSONEncoder().encode(result),
                  let jsonString = String(data: data, encoding: .utf8) else {
                return outputs.joined(separator: "\n\n")
            }
            return jsonString

        case .text:
            var seenCaseKeys = Set<String>()
            var uniqueLines: [String] = []
            for output in outputs {
                let lines = output.components(separatedBy: .newlines)
                for rawLine in lines {
                    var line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !line.isEmpty,
                          !line.hasPrefix("--- Part"),
                          !line.lowercased().contains("no legal cases were cited"),
                          !line.lowercased().contains("no cited cases"),
                          line.uppercased() != "NONE" else { continue }

                    if line.hasPrefix("Case Name —") || line.hasPrefix("Case Name -") {
                        line = line.replacingOccurrences(of: "Case Name —", with: "")
                                   .replacingOccurrences(of: "Case Name -", with: "")
                                   .trimmingCharacters(in: .whitespacesAndNewlines)
                    }

                    guard line.contains("—") || line.contains("-") || line.contains("SCC") || line.contains("AIR") else { continue }

                    let parts = line.components(separatedBy: "—")
                    let caseIdentifier = parts.first ?? line
                    let key = caseIdentifier.lowercased()
                        .replacingOccurrences(of: "vs.", with: "v.")
                        .replacingOccurrences(of: "versus", with: "v.")
                        .filter { $0.isLetter || $0.isNumber }
                    guard !key.isEmpty else { continue }

                    if !seenCaseKeys.contains(key) {
                        seenCaseKeys.insert(key)
                        uniqueLines.append(line)
                    }
                }
            }

            if uniqueLines.isEmpty {
                return "No cited cases found in this document."
            }
            return uniqueLines.joined(separator: "\n")
        }
    }

    /// 🚀 STEP 3: FIXED MAX_TOKENS AND TOKEN BUDGETING
    private func generateForChunk(
        container: ModelContainer, activePrompt: String, chunkText: String,
        chunkLabel: String?, thinkingEnabled: Bool
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
                        "thinking_budget": 128
                    ]
                )
            )

            var generateParameters = GenerateParameters(temperature: 0.0)
            generateParameters.topP = 0.95
            generateParameters.repetitionPenalty = 1.15
            // ⚡ Increased from 700 to 2048: Fixes the truncation bug in Batch 5
            generateParameters.maxTokens = 2048
            
            let stream = try MLXLMCommon.generate(
                input: input,
                parameters: generateParameters,
                context: context
            )

            let comesFromBuiltInRegistry = context.configuration.reasoningConfig != nil
            let isPrimedInside = thinkingEnabled && comesFromBuiltInRegistry

            let effectiveReasoningConfig: ReasoningConfig? = thinkingEnabled
                ? (context.configuration.reasoningConfig ?? Self.gemmaChannelReasoningConfig)
                : nil

            var emitter: ReasoningEventEmitter? = effectiveReasoningConfig.map {
                ReasoningEventEmitter(config: $0, primedInside: isPrimedInside)
            }

            var fullText = ""
            var thinkingText = ""
            var metrics = ChunkMetrics()

            func route(_ segments: [ReasoningEventEmitter.Segment]) {
                for segment in segments {
                    switch segment {
                    case .reasoning(let t): thinkingText += t
                    case .response(let t): fullText += t
                    }
                }
            }

            for await generation in stream {
                switch generation {
                case .chunk(let text):
                    if emitter != nil {
                        route(emitter!.process(text))
                    } else {
                        fullText += text
                    }

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
            if emitter != nil {
                route(emitter!.finalize())
            }

            metrics.thinkingChars = thinkingText.count
            metrics.responseChars = fullText.count

            if !thinkingText.isEmpty {
                print(" Thinking (\(thinkingText.count) chars, hidden from UI):\n\(thinkingText)")
            }

            return ChunkResult(text: fullText, metrics: metrics)
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
