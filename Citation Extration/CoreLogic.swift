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
    var title: String // First chat/prompt text
    var pdfText: String
    var extractedOutput: String
    var outputFormat: OutputFormat// JSON or Text — remembers which format this was generated in
    var generationTimeText: String

}

// MARK: - Structured Extraction Schema
// Swift equivalent of the Zod schema — the model must return JSON in this exact shape
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

    /// Lenient init — small local models sometimes send "null" as a quoted string
    /// instead of real JSON null, or send year as a string ("2023"). This handles
    /// all of that gracefully so parsing never fails.
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
            return Int(trimmed)  // "null" or any non-numeric string becomes nil here
        }
        return nil
    }
}

struct DocumentExtractionResult: Codable {
    let citedCases: [CitedCase]

    /// The model's raw output sometimes comes wrapped in ```json ... ``` fences or with
    /// extra surrounding text — this cleans that up and extracts strict JSON.
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

        // Take only the portion from the first '{' to the last '}' — the model
        // sometimes adds extra preamble/explanation before or after the JSON.
        guard let firstBrace = text.firstIndex(of: "{"),
              let lastBrace = text.lastIndex(of: "}") else { return nil }
        text = String(text[firstBrace...lastBrace])

        guard let data = text.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(DocumentExtractionResult.self, from: data)
    }
}

enum ModelArchitectureType {
    case llm   // Text-only — loaded via MLXLLM / LLMModelFactory
    case vlm   // Vision-Language — loaded via MLXVLM / VLMModelFactory (works text-only too if no image is given)
}

enum LLMModel: String, CaseIterable, Identifiable {
    case gemma4 = "Gemma 4 E2B"
    case gemma3n = "Gemma 3n E2B"
    case qwen35 = "Qwen 3.5 2B"
    case qwen25 = "Qwen 2.5 1.5B"
    case addModel = "Add New Model..."

    var id: String { rawValue }

    // Verified repo IDs — confirmed directly from the mlx-swift-lm library source
    // (LLMModelFactory.swift / VLMModelFactory.swift)
    var hubRepoID: String? {
        switch self {
        case .gemma4:
            // Confirmed: registered in the VLMModelFactory registry ("gemma4" architecture) — requires MLXVLM
            return "mlx-community/gemma-4-e2b-it-4bit"
        case .gemma3n:
            // Confirmed: registered in LLMModelFactory — text-only, no extra package needed, verified working
            return "mlx-community/gemma-3n-E2B-it-lm-4bit"
        case .qwen35:
            // Confirmed: registered in LLMModelFactory with the "qwen3_5" architecture
            // (requires library version 3.31.4 or newer — run 'Update to Latest Package
            // Versions' in Package Dependencies if you get an unsupportedModelType error).
            return "mlx-community/Qwen3.5-2B-4bit"
        case .qwen25:
            // "qwen2" architecture — the very first verified working model, a reliable fallback
            return "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        case .addModel:
            return nil
        }
    }

    // Which factory (LLM or VLM) to use, depending on the model
    var architectureType: ModelArchitectureType {
        switch self {
        case .gemma4: return .vlm
        case .gemma3n: return .llm
        case .qwen35: return .llm
        case .qwen25: return .llm
        case .addModel: return .llm
        }
    }
}

// =================================================================================================
// 1. PDF TO TEXT CONVERSION LOGIC
// Answer: YES, the PDF is converted to text here.
// Note: The AI Model DOES NOT read the PDF file directly. Apple's native `PDFKit` reads the PDF,
// extracts the text page-by-page, and then this raw text is sent to the AI model.
// =================================================================================================
class PDFParser {
    /// Inserted between pages during extraction so downstream code (page-based chunking)
    /// can split the text back into individual pages. Very unlikely to appear naturally
    /// in real document text.
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

// MARK: - Manual Downloader / TokenizerLoader bridges
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

// Output format the user picks in the UI dropdown
enum OutputFormat: String, CaseIterable, Identifiable {
    case json = "JSON"
    case text = "Text"

    var id: String { rawValue }
}

// MARK: - Instrumentation
// These live at file scope rather than nested inside LLMManager on purpose: the class is
// @MainActor, and these values are constructed inside the non-isolated `@Sendable` closure
// passed to `ModelContainer.perform`. Nesting them would inherit MainActor isolation and
// make them unusable there — the same reason `gemmaChannelReasoningConfig` is `nonisolated`.

/// Whether a batch's output could be parsed back into structured cases.
enum ParseStatus: Sendable {
    case notApplicable      // Text mode — there is no JSON to parse
    case failed             // JSON mode, but parsing failed (malformed or truncated)
    case parsed(count: Int)
}

/// Per-batch timing and token telemetry.
///
/// This exists to answer two questions we were previously guessing at: how much of the
/// wall clock is prefill vs decode (which decides whether prefix caching or faster decode
/// is the optimisation worth doing), and whether any batch was silently truncated by the
/// token limit — which drops citations without surfacing any error to the user.
///
/// All of it comes free from `GenerateCompletionInfo` on the generation stream; none of it
/// costs an extra model or tokenizer pass.
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

    var totalSeconds: Double { prefillSeconds + decodeSeconds }

    /// Share of generated output that was thinking, measured in characters.
    /// Characters are a proxy: the stream reports an exact token count for the whole
    /// generation, but does not tell us which of those tokens were reasoning.
    var thinkingShare: Double {
        let total = thinkingChars + responseChars
        guard total > 0 else { return 0 }
        return Double(thinkingChars) / Double(total)
    }

    /// The exact generated-token count, apportioned by the thinking/response character split.
    var estimatedThinkingTokens: Int {
        Int((Double(generatedTokens) * thinkingShare).rounded())
    }

    var summaryLine: String {
        var parts: [String] = ["Batch \(label)"]
        parts.append(String(format: "%.2fs (prefill %.2fs / decode %.2fs)",
                            totalSeconds, prefillSeconds, decodeSeconds))
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

/// Generated text plus the telemetry for the call that produced it.
struct ChunkResult: Sendable {
    let text: String
    let metrics: ChunkMetrics
}

// MARK: - LLM Manager
@MainActor
class LLMManager: ObservableObject {
    @Published var isGenerating = false
    @Published var downloadProgress: Double = 0.0   // 0...1, while model weights are downloading
    @Published var statusText: String = ""
    
    // Variable to display generation time in the UI
    @Published var generationTimeText: String = ""

    let textOnlySystemPrompt = """
    You are an expert Legal AI Assistant specializing in Indian jurisprudence and legal document analysis.
    Your task is to carefully read through the given Indian court document and list out every legal case
    cited as a precedent or reference within it.

    For each cited case, on its own line, write: Case Name — Citation (if available) — one-line reason it was cited.
    Scan the ENTIRE document from beginning to end, including footnotes and any quoted passages from lower
    court/tribunal judgments. If a case is only cited inside a footnote or inside a quoted paragraph, still list it.

    If you cannot find any cited cases, say so explicitly and briefly explain what the document is about instead,
    so we can confirm you are reading the actual document content.

    Do not use JSON. Just write plain, readable text.
    """

    let jsonSystemPrompt = """
    You are an expert Legal AI Assistant specializing in Indian jurisprudence and legal document analysis.
    Your task is to carefully read through Indian court filing documents and extract every legal case cited as a precedent or reference.

    Follow these strict guidelines:
    1. Identify Case Names: Look for standard adversarial formats (e.g., "X v. Y", "X vs. Y", "In Re: X").
    2. Identify Citations: Look for standard Indian legal reporters and journals. Common examples include, but are not limited to:
       - SCC (Supreme Court Cases)
       - AIR (All India Reporter)
       - SCR (Supreme Court Reports)
       - SCALE, JT (Judgment Today)
       - High Court specific reporters (e.g., DLT, BomLR, KLT, MLJ)
       - Neutral citations (e.g., 2023/DHC/1234)
    3. Differentiate: Do not include the primary case (the case currently being heard/filed) in the list of cited cases, unless it is explicitly cited in a historical context within the document.
    4. Context: Briefly summarize the legal principle or reason why the case was cited based on the surrounding text.
    5. Missing Data: If a specific detail (like the year or court) is not mentioned in the text, return null for that field. Do not hallucinate information not present in the document.
    6. Thoroughness: Scan the ENTIRE document from beginning to end, including the last few paragraphs — do not stop early. Citations often appear later in the document (e.g., in a discussion, analysis, or "reasons for judgment" section), not just near the top.
    7. Nested/Quoted Citations: Citations may appear inside a quoted passage — for example, when this judgment quotes verbatim from a lower court's or tribunal's judgment (often in quotation marks or an indented block), and that quoted text itself references other cases. Extract those cited cases too, exactly as they appear, even if they are inside a quotation.
    8. Footnote Citations: Citations are sometimes given in footnotes (marked with superscript numbers like 1, 2, 3 in the body text) rather than inline in the main paragraph. Check footnote text at the bottom of pages for citations as well.

    OUTPUT FORMAT — this is mandatory:
    Respond with ONLY a single valid JSON object, and nothing else — no markdown fences (no ```),
    no explanations, no preamble, no text before or after the JSON, no closing remarks.

    The JSON must match this TypeScript type exactly:

    type CitedCase = {
      caseName: string;
      citation: string | null;   // use the real JSON null keyword when unknown, NEVER the text "null" as a string
      year: number | null;       // a plain number like 2019, or real JSON null — never a quoted string
      court: string | null;
      context: string | null;
    };
    type DocumentExtractionResult = {
      citedCases: CitedCase[];
    };

    Example of a correctly formatted response (follow this style exactly):
    {"citedCases":[{"caseName":"K.M. Nanavati v. State of Maharashtra","citation":"AIR 1962 SC 605","year":1962,"court":"Supreme Court of India","context":"Cited regarding the scope of judicial review of jury verdicts."},{"caseName":"State of Punjab v. Bhajan Kaur","citation":null,"year":null,"court":null,"context":"Referenced as a precedent for motor accident compensation without further detail given in the text."}]}

    If no cases are cited in the document, respond with exactly: {"citedCases":[]}
    """

    // Loaded models are cached so we don't re-download/re-load repeatedly
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

        loadedContainers[repoID] = container
        statusText = ""
        downloadProgress = 1.0
        return container
    }

    func generateStructuredOutput(pdfText: String, model: LLMModel, outputFormat: OutputFormat, thinkingEnabled: Bool = true) async throws -> String {
        guard let repoID = model.hubRepoID else {
            throw LLMError.noRepoConfigured
        }

        isGenerating = true
        generationTimeText = "Processing..." // Reset time text
        
        defer { isGenerating = false }

        let container = try await loadContainer(repoID: repoID, architectureType: model.architectureType)
        let activePrompt = outputFormat == .json ? jsonSystemPrompt : textOnlySystemPrompt

        // START TOTAL TIMER
        let totalStartTime = CFAbsoluteTimeGetCurrent()

        // =================================================================================================
        // BATCH PROCESSING (SERIAL — deliberately)
        //
        // Batches run one at a time. This is not a missed optimisation: `ModelContainer.perform`
        // takes an exclusive async lock (SerialAccessContainer -> AsyncMutex) held for the whole
        // duration of the call, so concurrent batches queue behind each other regardless. The
        // TaskGroup that used to live here produced zero parallelism and logs that claimed
        // otherwise. Real concurrency would require batched generation (one forward pass over N
        // sequences), which the high-level MLX API does not expose.
        // =================================================================================================

        let batches = Self.chunkByPages(pdfText)
        let isSingleBatch = batches.count == 1

        var chunkOutputs: [String] = []
        var allMetrics: [ChunkMetrics] = []

        for (index, batch) in batches.enumerated() {
            statusText = isSingleBatch
                ? "Processing document..."
                : "Processing batch \(index + 1) of \(batches.count) (\(batch.pageRangeLabel))..."

            // Single-batch documents get no batch label, so their prompt is identical
            // to what a whole-document run has always sent.
            let label: String? = isSingleBatch
                ? nil
                : "This is a batch of pages (\(batch.pageRangeLabel)) from a larger legal document, part \(index + 1) of \(batches.count) overall. Extract cited cases found in THIS BATCH only."

            print("▶️ [START] Batch \(index + 1)/\(batches.count) (\(batch.pageRangeLabel))...")

            let result = try await generateForChunk(
                container: container, activePrompt: activePrompt, chunkText: batch.text,
                chunkLabel: label, thinkingEnabled: thinkingEnabled
            )

            var metrics = result.metrics
            metrics.label = "\(index + 1)/\(batches.count) \(batch.pageRangeLabel)"

            // Record whether this batch's JSON actually parsed. A truncated or malformed
            // batch contributes zero cases to the merge and would otherwise vanish silently.
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
        if isSingleBatch {
            finalResult = chunkOutputs[0]
        } else {
            statusText = "Merging results from \(batches.count) page-batches..."
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

    /// One batch of consecutive pages, ready to send to the model as a single chunk.
    private struct PageBatch {
        let text: String
        let pageRangeLabel: String  // e.g. "pages 6–10"
    }

    /// Prints the aggregate picture for a run: where the time went, how much of it was
    /// thinking, and whether anything was truncated or failed to parse.
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

        print("""

        ═══════════ RUN SUMMARY (\(outputFormat.rawValue)) ═══════════
        Batches:        \(metrics.count)
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

    /// Splits the document strictly into batches of `pagesPerBatch` consecutive pages.
    /// No character limit is applied — 5 pages is always exactly 1 batch, however long
    /// those pages are. Watch the run summary for truncation warnings if pages are dense.
        private static func chunkByPages(_ text: String, pagesPerBatch: Int = 5) -> [PageBatch] {
            let pages = text.components(separatedBy: PDFParser.pageBreakMarker)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }

            // If no page markers are found, return the whole text as one batch
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

                // Directly append the 5 pages as ONE single batch
                batches.append(PageBatch(text: batchText, pageRangeLabel: rangeLabel))
                
                index = end
            }
            
            return batches
        }

    /// Merges the per-chunk outputs into one final answer.
    private static func mergeChunkOutputs(_ outputs: [String], outputFormat: OutputFormat) -> String {
        switch outputFormat {
        case .json:
            var merged: [CitedCase] = []
            var seenKeys = Set<String>()
            for output in outputs {
                guard let parsed = DocumentExtractionResult.parse(from: output) else { continue }
                for citedCase in parsed.citedCases {
                    let key = citedCase.caseName.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !key.isEmpty, !seenKeys.contains(key) else { continue }
                    seenKeys.insert(key)
                    merged.append(citedCase)
                }
            }
            let result = DocumentExtractionResult(citedCases: merged)
            guard let data = try? JSONEncoder().encode(result),
                  let jsonString = String(data: data, encoding: .utf8) else {
                return outputs.joined(separator: "\n\n")
            }
            return jsonString

        case .text:
            return outputs.enumerated()
                .map { index, text in "--- Part \(index + 1) ---\n\(text.trimmingCharacters(in: .whitespacesAndNewlines))" }
                .joined(separator: "\n\n")
        }
    }

    /// Runs one generation call for a single chunk of text (or the whole document, if not chunked).
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
                        additionalContext: ["enable_thinking": thinkingEnabled]
                    )
                )

                var generateParameters = GenerateParameters(temperature: 0.0)
                generateParameters.maxTokens = 4096

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
                        // Exact prefill/decode split and stop reason, straight from the
                        // generator — no extra tokenizer pass needed to measure this.
                        metrics.promptTokens = info.promptTokenCount
                        metrics.generatedTokens = info.generationTokenCount
                        metrics.prefillSeconds = info.promptTime
                        metrics.decodeSeconds = info.generateTime
                        switch info.stopReason {
                        case .stop:
                            metrics.stopReasonText = "stop"
                        case .length:
                            // Generation was cut off by maxTokens rather than finishing.
                            // With thinking on, reasoning tokens share this budget.
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
    // =========================================================================
        // 🌟 NEW FEATURE: GENERAL CHAT MODE (For questions like "What is photosynthesis?")
        // =========================================================================
        
      /*  let generalSystemPrompt = "You are a helpful, smart, and concise AI assistant. Answer the user's questions accurately."

        func askGeneralQuestion(question: String, model: LLMModel) async throws -> String {
            guard let repoID = model.hubRepoID else {
                throw LLMError.noRepoConfigured
            }

            isGenerating = true
            statusText = "Thinking..."
            defer { isGenerating = false }

            // 1. Load the model
            let container = try await loadContainer(repoID: repoID, architectureType: model.architectureType)

            // 2. Ask the question directly (No PDF chunking, No JSON formatting)
            let output = try await container.perform { context in
                let chatMessages: [Chat.Message] = [
                    Chat.Message(role: .system, content: generalSystemPrompt),
                    Chat.Message(role: .user, content: question)
                ]

                let input = try await context.processor.prepare(
                    input: UserInput(prompt: .chat(chatMessages))
                )

                // Temperature 0.6 rakhi hai taaki model thoda creative aur natural answer de
                var generateParameters = GenerateParameters(temperature: 0.6)
                generateParameters.maxTokens = 1024

                let stream = try MLXLMCommon.generate(
                    input: input,
                    parameters: generateParameters,
                    context: context
                )

                var fullText = ""
                for await generation in stream {
                    if case .chunk(let text) = generation {
                        fullText += text
                        // Agar aap UI mein live typing dikhana chahte hain, toh yahan update kar sakte hain
                    }
                }
                return fullText
            }

            statusText = "Done!"
            return output
        }*/

    enum LLMError: LocalizedError {
        case noRepoConfigured
        var errorDescription: String? {
            "No Hugging Face repo ID is set for this model. Add one in LLMModel.hubRepoID."
        }
    }
}
