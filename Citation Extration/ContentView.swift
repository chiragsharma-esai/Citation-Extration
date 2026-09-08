import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var llmManager = LLMManager()
    
    @State private var histories: [ChatHistory] = []
    @State private var selectedHistory: ChatHistory?
    
    @State private var selectedModel: LLMModel = .gemma4
    @State private var selectedOutputFormat: OutputFormat = .text
    @State private var thinkingEnabled: Bool = true
    @State private var preFilterEnabled: Bool = true
    
    @State private var uploadedPDFText: String? = nil
    @State private var uploadedFileName: String? = nil
    @State private var currentOutput: String = ""
    @State private var errorMessage: String? = nil

    @State private var activePDFText: String? = nil
    @State private var lastGeneratedFormat: OutputFormat? = nil
   
    @State private var currentThinking: String = ""
    @State private var isThinkingExpanded: Bool = true
    @State private var userQuestion: String = ""
    
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

                // New Chat
                Button(action: startNewChat) {
                    Label("New Chat", systemImage: "plus.circle.fill")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                .padding(8)
            }
            .navigationTitle("Chats")
            .onChange(of: selectedHistory) { oldValue, newValue in
                guard let newValue = newValue else { return }
                currentOutput = newValue.extractedOutput
                currentThinking = newValue.thinkingText
                activePDFText = newValue.pdfText
                lastGeneratedFormat = newValue.outputFormat
                selectedOutputFormat = newValue.outputFormat
                selectedModel = newValue.model
                thinkingEnabled = newValue.thinkingEnabled
                llmManager.generationTimeText = newValue.generationTimeText
                uploadedPDFText = nil
                errorMessage = nil
            }

        } detail: {
            // MARK: - Main Chat/Extraction Area
            VStack(spacing: 20) {
                
                // Top Bar: Model + Output Format + Thinking Selection
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

                    Picker("Output", selection: $selectedOutputFormat) {
                        ForEach(OutputFormat.allCases) { format in
                            Text(format.rawValue).tag(format)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 140)
                    .disabled(llmManager.isGenerating)
                    .onChange(of: selectedOutputFormat) { _, newFormat in
                        regenerateIfFormatChanged(to: newFormat)
                    }

                    Picker("Model", selection: $selectedModel) {
                        ForEach(LLMModel.allCases, id: \.self) { model in
                            Text(model.rawValue).tag(model)
                        }
                    }
                    .pickerStyle(.menu)
                    .frame(width: 200)
                    .padding()
                }

                if llmManager.isGenerating && !currentOutput.isEmpty {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text("Re-generating in \(selectedOutputFormat.rawValue) format...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                // MARK: - Output Area
                if !currentOutput.isEmpty || !currentThinking.isEmpty {
                    let parsedResult = DocumentExtractionResult.parse(from: currentOutput)

                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            
                            // COLLAPSIBLE THOUGHTS CARD
                            if thinkingEnabled && !currentThinking.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                VStack(alignment: .leading, spacing: 12) {
                                    Button(action: {
                                        withAnimation(.easeInOut(duration: 0.2)) {
                                            isThinkingExpanded.toggle()
                                        }
                                    }) {
                                        HStack(spacing: 8) {
                                            Image(systemName: "sparkles")
                                                .foregroundColor(Color(nsColor: .systemBlue))
                                                .font(.headline)
                                            
                                            Text("Thoughts")
                                                .font(.headline)
                                                .foregroundColor(.primary)
                                            
                                            Spacer()
                                            
                                            Image(systemName: isThinkingExpanded ? "chevron.up" : "chevron.down")
                                                .foregroundColor(.secondary)
                                                .font(.body)
                                        }
                                    }
                                    .buttonStyle(.plain)
                                    
                                    if isThinkingExpanded {
                                        VStack(alignment: .leading, spacing: 12) {
                                            Divider()
                                                .background(Color.secondary.opacity(0.15))
                                            
                                            Text(currentThinking)
                                                .font(.body)
                                                .foregroundColor(.secondary)
                                                .lineSpacing(4)
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                            
                                            Divider()
                                                .background(Color.secondary.opacity(0.15))
                                            
                                            Button(action: {
                                                withAnimation(.easeInOut(duration: 0.2)) {
                                                    isThinkingExpanded = false
                                                }
                                            }){
                                                HStack {
                                                    Text("Collapse to hide model thoughts")
                                                        .font(.subheadline)
                                                        .foregroundColor(.secondary)
                                                    Spacer()
                                                    Image(systemName: "chevron.up")
                                                        .font(.caption)
                                                        .foregroundColor(.secondary)
                                                }
                                            }
                                            .buttonStyle(.plain)
                                        }
                                    }
                                }
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(Color(nsColor: .windowBackgroundColor))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
                                        )
                                )
                            }
                            
                            // EXECUTION TIME DISPLAY
                            if !llmManager.generationTimeText.isEmpty && !currentOutput.isEmpty {
                                let timeString = llmManager.generationTimeText
                                    .replacingOccurrences(of: "Time taken: ", with: "")
                                    .replacingOccurrences(of: " seconds", with: "s")
                                    .components(separatedBy: " ").first ?? ""
                                
                                Text(timeString)
                                    .font(.system(.subheadline, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .padding(.top, 4)
                            }
                            
                            // Response Output
                            if !currentOutput.isEmpty {
                                if let parsedResult, !parsedResult.citedCases.isEmpty {
                                    VStack(alignment: .leading, spacing: 12) {
                                        ForEach(parsedResult.citedCases) { citedCase in
                                            VStack(alignment: .leading, spacing: 4) {
                                                Text(citedCase.caseName).font(.headline)
                                                if let citation = citedCase.citation { Text(citation).font(.subheadline).foregroundColor(.secondary) }
                                                HStack(spacing: 12) {
                                                    if let year = citedCase.year { Text("Year: \(String(year))") }
                                                    if let court = citedCase.court { Text("Court: \(court)") }
                                                }
                                                .font(.caption).foregroundColor(.secondary)
                                                if let context = citedCase.context { Text(context).font(.body).padding(.top, 2) }
                                            }
                                            .padding()
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .background(Color(NSColor.textBackgroundColor))
                                            .cornerRadius(10)
                                        }
                                    }
                                } else if parsedResult != nil {
                                    Text("No cited cases found in this document.")
                                        .foregroundColor(.secondary).padding()
                                } else {
                                    Text(currentOutput)
                                        .padding()
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color(NSColor.textBackgroundColor))
                                        .cornerRadius(10)
                                }
                            }
                        }
                        .padding()
                    }
                    .padding(.horizontal)
                    
                    // Export Buttons
                    if !currentOutput.isEmpty {
                        HStack {
                            Spacer()
                            Button(action: { ExportManager.saveAsCSV(citedCases: parsedResult?.citedCases, rawContent: currentOutput) }) {
                                Label("Save in Excel", systemImage: "tablecells")
                            }
                            Button(action: { ExportManager.saveAsPDF(citedCases: parsedResult?.citedCases, rawContent: currentOutput) }) {
                                Label("Save in PDF", systemImage: "doc.text")
                            }
                        }
                        .padding(.horizontal)
                    }
                }
                
                Spacer()
                
                // MARK: - Upload Area & Bottom Loading
                if currentOutput.isEmpty && currentThinking.isEmpty {
                    VStack(spacing: 20) {
                        
                        // Option: PDF Upload
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

                        // Bottom Status, Spinner & Download Progress Bar
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
                    }
                    .padding()
                    .frame(maxWidth: 600)
                }
            }
            .navigationTitle("Citation Extraction")
        }
    }
    
    // MARK: - Actions
    private func startNewChat() {
        selectedHistory = nil
        uploadedPDFText = nil
        uploadedFileName = nil
        currentOutput = ""
        currentThinking = ""
        isThinkingExpanded = true
        errorMessage = nil
        activePDFText = nil
        lastGeneratedFormat = nil
        userQuestion = ""
    }
    
    private func regenerateIfFormatChanged(to newFormat: OutputFormat) {
        guard newFormat != lastGeneratedFormat else { return }
        guard let text = activePDFText, !currentOutput.isEmpty else { return }

        errorMessage = nil
        Task {
            do {
                self.currentOutput = ""
                self.currentThinking = ""
                self.isThinkingExpanded = true
                
                var tempOutput = ""
                
                let result = try await llmManager.generateStructuredOutput(
                    pdfText: text,
                    model: selectedModel,
                    outputFormat: newFormat,
                    thinkingEnabled: thinkingEnabled,
                    preFilterEnabled: preFilterEnabled,
                    onThinkingToken: { token in
                        self.currentThinking += token
                    },
                    onToken: { token in
                        tempOutput += token
                    }
                )
                
                if result.responseText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.errorMessage = "Model returned empty output."
                    return
                }
                
                self.currentThinking = result.thinkingText
                self.currentOutput = result.responseText
                self.lastGeneratedFormat = newFormat

                if let selectedHistory, let index = histories.firstIndex(where: { $0.id == selectedHistory.id }) {
                    histories[index].extractedOutput = result.responseText
                    histories[index].thinkingText = result.thinkingText
                    histories[index].thinkingEnabled = thinkingEnabled
                    histories[index].outputFormat = newFormat
                    self.selectedHistory = histories[index]
                }
            } catch {
                self.errorMessage = "Format switch failed: \(error.localizedDescription)"
            }
        }
    }

    private func selectPDF() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.pdf]
        panel.allowsMultipleSelection = false

        if panel.runModal() == .OK, let url = panel.url {
            if let text = PDFParser.extractText(from: url), !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                self.uploadedPDFText = text
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
                self.isThinkingExpanded = true
                
                var tempOutput = ""
                
                let result = try await llmManager.generateStructuredOutput(
                    pdfText: text,
                    model: selectedModel,
                    outputFormat: selectedOutputFormat,
                    thinkingEnabled: thinkingEnabled,
                    preFilterEnabled: preFilterEnabled,
                    onThinkingToken: { token in
                        self.currentThinking += token
                    },
                    onToken: { token in
                        tempOutput += token
                    }
                )

                if result.responseText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.errorMessage = "Model returned empty output."
                    return
                }

                self.currentThinking = result.thinkingText
                self.currentOutput = result.responseText
                self.activePDFText = text
                self.lastGeneratedFormat = selectedOutputFormat
                self.isThinkingExpanded = true

                let title = uploadedFileName ?? "Untitled Extraction"
                let newHistory = ChatHistory(
                    title: title,
                    pdfText: text,
                    extractedOutput: result.responseText,
                    outputFormat: selectedOutputFormat,
                    generationTimeText: llmManager.generationTimeText,
                    thinkingText: result.thinkingText,
                    thinkingEnabled: thinkingEnabled,
                    model: selectedModel
                )
                self.histories.insert(newHistory, at: 0)
                self.selectedHistory = newHistory

                self.uploadedPDFText = nil
                self.uploadedFileName = nil

            } catch {
                self.errorMessage = "Extraction failed: \(error.localizedDescription)"
            }
        }
    }
}
