import SwiftUI
import PDFKit
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var llmManager = LLMManager()

    @State private var histories: [ChatHistory] = []
    @State private var selectedHistory: ChatHistory?

    @State private var selectedModel: LLMModel = .gemma4
    @State private var thinkingEnabled: Bool = true
    @State private var preFilterEnabled: Bool = true

    @State private var uploadedPDFText: String? = nil
    @State private var uploadedPDFURL: URL? = nil
    @State private var uploadedFileName: String? = nil
    @State private var currentOutput: String = ""
    @State private var errorMessage: String? = nil
    @State private var activePDFText: String? = nil

    @State private var currentThinking: String = ""
    @State private var isThinkingExpanded: Bool = false

    @State private var parsedCases: [CitedCase] = []
    @State private var selectedCaseID: UUID? = nil

    @State private var pdfDocumentForViewer: PDFDocument? = nil
    @State private var highlightText: String? = nil
    @State private var highlightCitation: String? = nil
    @State private var highlightPage: Int? = nil

    var body: some View {
        NavigationSplitView {
            // MARK: - Left Sidebar (History)
            VStack(spacing: 0) {
                List(selection: $selectedHistory) {
                    ForEach(histories) { history in
                        Text(history.title)
                            .font(.headline)
                            .lineLimit(1)
                            .tag(history)
                    }
                }
                .listStyle(.sidebar)

                Divider()

                Button(action: startNewChat) {
                    Label("New Chat", systemImage: "plus.circle.fill")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                .padding(8)
            }
            .navigationTitle("Chats")
            .onChange(of: selectedHistory) { _, newValue in
                guard let newValue = newValue else { return }
                currentOutput = newValue.extractedOutput
                currentThinking = newValue.thinkingText
                activePDFText = newValue.pdfText
                selectedModel = newValue.model
                thinkingEnabled = newValue.thinkingEnabled
                llmManager.generationTimeText = newValue.generationTimeText
                uploadedPDFText = nil
                errorMessage = nil
                selectedCaseID = nil
                highlightText = nil
                highlightCitation = nil
                highlightPage = nil

                if let parsed = DocumentExtractionResult.parse(from: newValue.extractedOutput) {
                    parsedCases = parsed.citedCases
                } else {
                    parsedCases = []
                }

                if let url = newValue.pdfURL {
                    pdfDocumentForViewer = PDFDocument(url: url)
                } else {
                    pdfDocumentForViewer = nil
                }
            }

        } detail: {
            // MARK: - Main Area
            VStack(spacing: 0) {

                // Top Bar
                HStack(spacing: 12) {
                    if !llmManager.isGenerating && !llmManager.generationTimeText.isEmpty {
                        Text(llmManager.generationTimeText)
                            .font(.caption)
                            .foregroundColor(.green)
                            .bold()
                    }

                    Spacer()

                    Toggle("Pre-filter", isOn: $preFilterEnabled)
                        .toggleStyle(.switch)
                        .disabled(llmManager.isGenerating)
                        .frame(width: 130)
                        .help("Send only passages around citation-shaped text instead of whole pages")

                    Toggle("Thinking", isOn: $thinkingEnabled)
                        .toggleStyle(.switch)
                        .disabled(llmManager.isGenerating)
                        .frame(width: 120)

                    Picker("Model", selection: $selectedModel) {
                        ForEach(LLMModel.allCases, id: \.self) { model in
                            Text(model.rawValue).tag(model)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 200)
                    .padding()
                }
                .padding(.horizontal)

                Divider()

                // MARK: - Content Area
                if !parsedCases.isEmpty || !currentThinking.isEmpty {
                    resultsView
                } else if currentOutput.isEmpty && currentThinking.isEmpty {
                    uploadView
                } else {
                    // Generating state — show progress
                    VStack(spacing: 16) {
                        Spacer()
                        if llmManager.isGenerating {
                            ProgressView()
                            if !llmManager.statusText.isEmpty {
                                Text(llmManager.statusText)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            if llmManager.downloadProgress > 0 && llmManager.downloadProgress < 1 {
                                ProgressView(value: llmManager.downloadProgress, total: 1.0)
                                    .progressViewStyle(.linear)
                                    .frame(width: 250)
                            }
                        }
                        Spacer()
                    }
                }
            }
            .navigationTitle("Citation Extraction")
        }
    }

    // MARK: - Results View (Table + PDF Side-by-Side)
    @ViewBuilder
    private var resultsView: some View {
        VStack(spacing: 0) {
            // Collapsible Thinking Card
            if thinkingEnabled && !currentThinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                thinkingCard
                    .padding(.horizontal, 16)
                    .padding(.top, 12)
            }

            // Execution time + count
            HStack {
                if !llmManager.generationTimeText.isEmpty && !currentOutput.isEmpty {
                    let timeString = llmManager.generationTimeText
                        .replacingOccurrences(of: "Time taken: ", with: "")
                        .replacingOccurrences(of: " seconds", with: "s")
                        .components(separatedBy: " ").first ?? ""
                    Text(timeString)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }

                if !parsedCases.isEmpty {
                    let citedCount = parsedCases.filter { !$0.isSelfReference }.count
                    Text("\(citedCount) cited case\(citedCount == 1 ? "" : "s") found")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if parsedCases.contains(where: { $0.isSelfReference }) {
                        HStack(spacing: 4) {
                            Circle()
                                .fill(Color.accentColor)
                                .frame(width: 6, height: 6)
                            Text("first row is this document's own case")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                Spacer()

                Button(action: { ExportManager.saveAsCSV(citedCases: parsedCases) }) {
                    Label("Export CSV", systemImage: "tablecells")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(parsedCases.isEmpty)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            Divider()

            // Side-by-side: Table + PDF
            if pdfDocumentForViewer != nil {
                HSplitView {
                    citationTableView
                        .frame(minWidth: 420, idealWidth: 520)

                    PDFViewerPanel(
                        pdfDocument: pdfDocumentForViewer,
                        highlightText: highlightText,
                        highlightCitation: highlightCitation,
                        highlightPage: highlightPage
                    )
                    .frame(minWidth: 350, idealWidth: 450)
                }
            } else {
                citationTableView
            }
        }
        .onChange(of: selectedCaseID) { _, newID in
            guard let newID,
                  let citedCase = parsedCases.first(where: { $0.id == newID }) else {
                highlightText = nil
                highlightCitation = nil
                highlightPage = nil
                return
            }
            
            // CITATION HIGHLIGHT LOGIC:
            // Prioritize highlighting the citation (e.g., "2007 1 SSC 789") in the document viewer.
            // Fall back to the case party name only if the citation is unavailable or marked as "N/A".
            let cleanCitation = citedCase.citation?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let cleanCitation, !cleanCitation.isEmpty && cleanCitation.uppercased() != "N/A" {
                highlightText = cleanCitation
            } else {
                highlightText = citedCase.caseName
            }

            highlightCitation = citedCase.citation
            highlightPage = citedCase.pageNumber
        }
    }

    // MARK: - Citation Table (Citation First, Party Names Second)
    @ViewBuilder
    private var citationTableView: some View {
        if parsedCases.isEmpty {
            VStack {
                Spacer()
                Text("No cited cases found in this document.")
                    .foregroundColor(.secondary)
                Spacer()
            }
        } else {
            Table(parsedCases, selection: $selectedCaseID) {
                // 1. Citation Column (Displayed first, includes "THIS DOCUMENT" badge for self-reference)
                TableColumn("Citation") { c in
                    HStack(spacing: 6) {
                        if c.isSelfReference {
                            Text("THIS DOCUMENT")
                                .font(.system(size: 9, weight: .bold))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.accentColor.opacity(0.18)))
                                .foregroundColor(.accentColor)
                                .fixedSize()
                        }
                        Text(c.citation ?? "N/A")
                            .foregroundColor(c.citation != nil ? (c.isSelfReference ? .accentColor : .primary) : .secondary)
                            .fontWeight(c.isSelfReference ? .semibold : .regular)
                            .lineLimit(1)
                    }
                }
                .width(min: 140, ideal: 200)

                // 2. Party Names Column (Displayed second)
                TableColumn("Party Names") { c in
                    Text(c.caseName)
                        .lineLimit(2)
                        .foregroundColor(c.isSelfReference ? .accentColor : .primary)
                        .fontWeight(c.isSelfReference ? .semibold : .regular)
                        .help(c.isSelfReference
                              ? "This is the document's own case, not a cited precedent"
                              : c.caseName)
                }
                .width(min: 160, ideal: 220)

                // 3. Court Column
                TableColumn("Court") { c in
                    Text(c.court ?? "—")
                        .foregroundColor(c.court != nil ? .primary : .secondary)
                        .lineLimit(1)
                }
                .width(min: 70, ideal: 90)

                // 4. Context Column
                TableColumn("Context") { c in
                    Text(c.context ?? "—")
                        .foregroundColor(c.context != nil ? .primary : .secondary)
                        .lineLimit(3)
                        .help(c.context ?? "")
                }
                .width(min: 140, ideal: 180)

                // 5. Page Column
                TableColumn("Page") { c in
                    Text(c.pageNumber.map(String.init) ?? "—")
                        .monospacedDigit()
                }
                .width(min: 45, ideal: 55)
            }
            .tableStyle(.inset(alternatesRowBackgrounds: true))
        }
    }

    // MARK: - Thinking Card
    private var thinkingCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    isThinkingExpanded.toggle()
                }
            }) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundColor(Color(nsColor: .systemBlue))
                        .font(.subheadline)
                    Text("Thoughts")
                        .font(.subheadline).bold()
                        .foregroundColor(.primary)
                    Spacer()
                    Image(systemName: isThinkingExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
            }
            .buttonStyle(.plain)

            if isThinkingExpanded {
                Divider()
                Text(currentThinking)
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .lineSpacing(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(maxHeight: 150)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .windowBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                )
        )
    }

    // MARK: - Upload View
    private var uploadView: some View {
        VStack(spacing: 20) {
            Spacer()

            if uploadedPDFText == nil {
                Button(action: selectPDF) {
                    VStack {
                        Image(systemName: "doc.badge.plus").font(.largeTitle)
                        Text("File Upload").font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(40)
                    .background(RoundedRectangle(cornerRadius: 12).stroke(style: StrokeStyle(lineWidth: 2, dash: [5])).foregroundColor(.gray))
                }
                .buttonStyle(.plain)
                .disabled(llmManager.isGenerating)
            } else {
                Text("File Uploaded Successfully!").foregroundColor(.green)
                Button(action: extractCitations) {
                    if llmManager.isGenerating {
                        ProgressView().scaleEffect(0.8)
                    } else {
                        Text("Extract").bold()
                    }
                }
                .buttonStyle(.borderedProminent).controlSize(.large).disabled(llmManager.isGenerating)
            }

            if llmManager.isGenerating && !llmManager.statusText.isEmpty {
                VStack(spacing: 10) {
                    Text(llmManager.statusText)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    if llmManager.downloadProgress > 0 && llmManager.downloadProgress < 1 {
                        ProgressView(value: llmManager.downloadProgress, total: 1.0)
                            .progressViewStyle(.linear)
                            .frame(width: 250)
                    }
                }
            }

            if let errorMessage {
                Text(errorMessage).font(.caption).foregroundColor(.red).multilineTextAlignment(.center).frame(maxWidth: 500)
            }

            Spacer()
        }
        .padding()
        .frame(maxWidth: 600)
    }

    // MARK: - Actions
    private func startNewChat() {
        selectedHistory = nil
        uploadedPDFText = nil
        uploadedPDFURL = nil
        uploadedFileName = nil
        currentOutput = ""
        currentThinking = ""
        isThinkingExpanded = false
        errorMessage = nil
        activePDFText = nil
        parsedCases = []
        selectedCaseID = nil
        pdfDocumentForViewer = nil
        highlightText = nil
        highlightCitation = nil
        highlightPage = nil
    }

    private func selectPDF() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            if let text = PDFParser.extractText(from: url), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                self.uploadedPDFText = text
                self.uploadedPDFURL = url
                self.uploadedFileName = url.deletingPathExtension().lastPathComponent
                self.errorMessage = nil
            } else {
                self.errorMessage = "Could not extract text from PDF."
            }
        }
    }

    private func extractCitations() {
        guard let text = uploadedPDFText else { return }
        errorMessage = nil

        Task {
            do {
                self.currentOutput = ""
                self.currentThinking = ""
                self.isThinkingExpanded = false
                self.parsedCases = []
                self.selectedCaseID = nil

                let result = try await llmManager.generateStructuredOutput(
                    pdfText: text,
                    model: selectedModel,
                    thinkingEnabled: thinkingEnabled,
                    preFilterEnabled: preFilterEnabled,
                    onThinkingToken: { token in
                        self.currentThinking += token
                    },
                    onToken: { _ in }
                )

                if result.responseText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.errorMessage = "Model returned empty output."
                    return
                }

                self.currentThinking = result.thinkingText
                self.currentOutput = result.responseText
                self.activePDFText = text

                if let parsed = DocumentExtractionResult.parse(from: result.responseText) {
                    self.parsedCases = parsed.citedCases
                }

                if let url = self.uploadedPDFURL {
                    self.pdfDocumentForViewer = PDFDocument(url: url)
                }

                let title = uploadedFileName ?? "Untitled Extraction"
                let newHistory = ChatHistory(
                    title: title,
                    pdfText: text,
                    pdfURL: uploadedPDFURL,
                    extractedOutput: result.responseText,
                    generationTimeText: llmManager.generationTimeText,
                    thinkingText: result.thinkingText,
                    thinkingEnabled: thinkingEnabled,
                    model: selectedModel
                )
                self.histories.insert(newHistory, at: 0)
                self.selectedHistory = newHistory

                self.uploadedPDFText = nil
                self.uploadedPDFURL = nil
                self.uploadedFileName = nil

            } catch {
                self.errorMessage = "Extraction failed: \(error.localizedDescription)"
            }
        }
    }
}
