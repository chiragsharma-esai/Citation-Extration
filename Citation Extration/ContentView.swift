////  ContentView.swift
//  Citation Extration
//
//  Created by Mac Neo on 07/08/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var llmManager = LLMManager()
    
    @State private var histories: [ChatHistory] = []
    @State private var selectedHistory: ChatHistory?
    
    @State private var selectedModel: LLMModel = .gemma4
    @State private var selectedOutputFormat: OutputFormat = .text
    @State private var thinkingEnabled: Bool = true
    // Regex pre-filter: send the model only the passages around citation-shaped text.
    // Exposed as a toggle so it can be A/B'd against the full-text baseline on the
    // same document — it trades a recall risk for speed and must be measured, not assumed.
    @State private var preFilterEnabled: Bool = true
    
    @State private var uploadedPDFText: String? = nil
    @State private var uploadedFileName: String? = nil
    @State private var currentOutput: String = ""
    @State private var errorMessage: String? = nil

    @State private var activePDFText: String? = nil
    @State private var lastGeneratedFormat: OutputFormat? = nil
   

    //  NEW: State variable for General Chat Question
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
            .onChange(of: selectedHistory) { _, newValue in
                guard let newValue else { return }
                currentOutput = newValue.extractedOutput
                activePDFText = newValue.pdfText
                lastGeneratedFormat = newValue.outputFormat
                selectedOutputFormat = newValue.outputFormat
                uploadedPDFText = nil
                errorMessage = nil
            }

        } detail: {
            // MARK: - Main Chat/Extraction Area
            VStack(spacing: 20) {
                
                // Top Bar: Model + Output Format + Thinking Selection
                HStack {
                    //  NEW: Display Generation Time
                    if !llmManager.generationTimeText.isEmpty {
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
                if !currentOutput.isEmpty {
                    let parsedResult = DocumentExtractionResult.parse(from: currentOutput)

                    ScrollView {
                        if let parsedResult, !parsedResult.citedCases.isEmpty {
                            // Structured JSON parsed successfully
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
                            .padding()
                        } else if parsedResult != nil {
                            Text("No cited cases found in this document.")
                                .foregroundColor(.secondary).padding()
                        } else {
                            // General Chat Answer / Streaming Text will be displayed here!
                            Text(currentOutput)
                                .padding()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(NSColor.textBackgroundColor))
                                .cornerRadius(10)
                        }
                    }
                    .padding(.horizontal)
                    
                    // Export Buttons
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
                
                Spacer()
                
                // MARK: - Upload & Extract Area OR General Chat
                if currentOutput.isEmpty {
                    VStack(spacing: 20) {
                        
                        // --- OPTION 1: PDF UPLOAD ---
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
                        } else {
                            Text("File Uploaded Successfully!").foregroundColor(.green)
                            Button(action: extractCitations) {
                                if llmManager.isGenerating { ProgressView().scaleEffect(0.8) } else { Text("Extract").bold() }
                            }
                            .buttonStyle(.borderedProminent).controlSize(.large).disabled(llmManager.isGenerating)
                        }
                        
                        // --- DIVIDER ---
                     /*===   if uploadedPDFText == nil {
                            HStack {
                                VStack { Divider() }
                                Text("OR ASK A QUESTION").font(.caption).foregroundColor(.secondary)
                                VStack { Divider() }
                            }
                            .padding(.vertical, 10)
                            
                            // --- OPTION 2: GENERAL CHAT ---
                            HStack {
                                TextField("E.g., What is photosynthesis?", text: $userQuestion)
                                    .textFieldStyle(.roundedBorder)
                                    .disabled(llmManager.isGenerating)
                                    .onSubmit { askGeneralQuestionAction() }
                                
                                Button(action: askGeneralQuestionAction) {
                                    if llmManager.isGenerating {
                                        ProgressView().scaleEffect(0.7)
                                    } else {
                                        Text("Ask AI")
                                    }
                                }
                                .buttonStyle(.borderedProminent)
                                .disabled(userQuestion.isEmpty || llmManager.isGenerating)
                            }
                        }===*/

                        // Status & Errors
                        if llmManager.isGenerating && !llmManager.statusText.isEmpty {
                            Text(llmManager.statusText).font(.caption).foregroundColor(.secondary)
                            if llmManager.downloadProgress > 0 && llmManager.downloadProgress < 1 {
                                ProgressView(value: llmManager.downloadProgress).frame(width: 200)
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
        errorMessage = nil
        activePDFText = nil
        lastGeneratedFormat = nil
        userQuestion = ""
    }

    //  NEW: Action for General Chat
 /*===   private func askGeneralQuestionAction() {
        guard !userQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let question = userQuestion
        errorMessage = nil
        
        Task {
            do {
                // Calls the new function in LLMManager
                let output = try await llmManager.askGeneralQuestion(question: question, model: selectedModel)
                
                self.currentOutput = output
                self.activePDFText = nil // No PDF for this chat
                self.lastGeneratedFormat = .text
                
                // Save to History
                let newHistory = ChatHistory(title: question, pdfText: "", extractedOutput: output, outputFormat: .text)
                self.histories.insert(newHistory, at: 0)
                self.selectedHistory = newHistory
                
                self.userQuestion = "" // Clear text field
            } catch {
                self.errorMessage = "Error: \(error.localizedDescription)"
            }
        }
    }
==========================*/

    private func regenerateIfFormatChanged(to newFormat: OutputFormat) {
        guard newFormat != lastGeneratedFormat else { return }
        guard let text = activePDFText, !currentOutput.isEmpty else { return }

        errorMessage = nil
        self.currentOutput = "" //  NEW: Clear the canvas so streaming writes on an empty screen
        
        Task {
            do {
                // [OLD CODE]
                // let output = try await llmManager.generateStructuredOutput(pdfText: text, model: selectedModel, outputFormat: newFormat, thinkingEnabled: thinkingEnabled, preFilterEnabled: preFilterEnabled)
                
                // [NEW CODE] Appends live tokens as they are produced in real-time
                let output = try await llmManager.generateStructuredOutput(
                    pdfText: text,
                    model: selectedModel,
                    outputFormat: newFormat,
                    thinkingEnabled: thinkingEnabled,
                    preFilterEnabled: preFilterEnabled
                ) { token in
                    self.currentOutput += token // Append tokens dynamically to the UI view
                }
                
                if output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.errorMessage = "Model returned empty output."
                    return
                }
                self.currentOutput = output
                self.lastGeneratedFormat = newFormat

                if let selectedHistory, let index = histories.firstIndex(where: { $0.id == selectedHistory.id }) {
                    histories[index].extractedOutput = output
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
        self.currentOutput = "" //  NEW: Clear the canvas so streaming writes on an empty screen

        Task {
            do {
                // [OLD CODE]
                // let output = try await llmManager.generateStructuredOutput(pdfText: text, model: selectedModel, outputFormat: selectedOutputFormat, thinkingEnabled: thinkingEnabled, preFilterEnabled: preFilterEnabled)

                // [NEW CODE] Appends live tokens as they are produced in real-time
                let output = try await llmManager.generateStructuredOutput(
                    pdfText: text,
                    model: selectedModel,
                    outputFormat: selectedOutputFormat,
                    thinkingEnabled: thinkingEnabled,
                    preFilterEnabled: preFilterEnabled
                ) { token in
                    self.currentOutput += token // 👈 Append tokens dynamically to the UI view
                }

                if output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.errorMessage = "Model returned empty output."
                    return
                }

                self.currentOutput = output
                self.activePDFText = text
                self.lastGeneratedFormat = selectedOutputFormat

                let title = uploadedFileName ?? "Untitled Extraction"
                let newHistory = ChatHistory(title: title, pdfText: text, extractedOutput: output, outputFormat: selectedOutputFormat,  generationTimeText: llmManager.generationTimeText
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
