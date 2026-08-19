import Foundation

// =================================================================================================
// CITATION PRE-FILTER
//
// A cheap, deterministic scan that finds citation-SHAPED text so the model is only sent the
// passages that could contain a citation, instead of every page in full.
//
// This does NOT extract citations — the LLM still does that. This only decides what the LLM
// gets to read. It is therefore tuned for RECALL: a false positive costs a little speed, a
// false negative loses a citation. When in doubt it keeps the text.
//
// The reporter and neutral-citation vocabularies below are not invented — they are the distinct
// values actually stored in the `ilocase42` index (355 ComparativeCitations.publisher values and
// 39 NeutralCitation.NC_abvr values), reduced to base tokens and filtered as described on each
// list. Matching is on the publisher token alone, bounded by non-letters, with an optional
// parenthesised court suffix — deliberately not a full citation grammar.
// =================================================================================================

enum CitationScanner {

    /// Sentences kept either side of a match. 3 is a reasonable default: enough to carry the
    /// party names and the "relied on / distinguished" verb that usually sit in the neighbouring
    /// sentence, without pulling in a whole page.
    static let defaultSentenceWindow = 3

    // MARK: - Vocabularies

    /// Reporter/publisher base tokens, harvested from `ComparativeCitations.publisher`.
    ///
    /// Stored values like "AIR(SC)", "CTR(Mad)" and "AllMR(Cr)" are reduced to their base
    /// ("AIR", "CTR", "AllMR") because the optional parenthesised suffix in the pattern covers
    /// every court variant generically. Values seen fewer than 10 times were dropped as noise,
    /// as were tokens that are ordinary words or court abbreviations ("SC", "AC", "All", "Kar",
    /// "Comp", …) — those fire constantly in a judgment and would defeat the point of filtering.
    /// Add any back here if a real citation is being missed.
    private static let reporterTokens: [String] = [
        "ELT", "AIR", "ITR", "SCC", "CRLJ", "CTR", "TAXMAN", "LLJ",
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
    ]

    /// Neutral-citation abbreviations, harvested from `NeutralCitation.NC_abvr` (all 38 non-empty
    /// values). These appear as "2024 INSC 835", "2023:DHC:1234" or "2023/DHC/1234"; matching the
    /// abbreviation alone covers every separator style.
    private static let neutralTokens: [String] = [
        "APHC", "KER", "KHC", "RJ-JD", "CGHC", "INSC", "MPHC-JBP", "GUJHC",
        "HHC", "MPHC-IND", "MPHC-GWL", "RJ-JP", "DHC", "BHC-NAG", "MHC", "BHC-AUG",
        "PHHC", "AHC", "GAU-AS", "KHC-D", "MLHC", "KHC-K", "AHC-LKO", "BHC-OS",
        "BHC-GOA", "CHC-AS", "SHC", "THC", "CHC-OS", "JHHC", "BHC-KOL", "CHC-PB",
        "BHC-AS", "JKLHC-JMU", "UHC", "JKLHC-SGR", "OHC", "CHC-JP",
    ]

    /// Whether to also treat a bare "X v. Y" as a candidate.
    ///
    /// Kept ON because a case cited by party name with no reporter ("Excel Wear v. Union of
    /// India") carries no publisher token at all, and would otherwise be invisible to this
    /// filter — a silent recall loss in exactly the situation the model handles well. Set to
    /// false to filter on publishers only.
    static var includePartyMarker = true

    // MARK: - Pattern

    /// One alternation over both vocabularies, bounded by non-letters, with an optional
    /// parenthesised court suffix.
    ///
    /// - `(?<![A-Za-z])` / `(?![A-Za-z])` mean "not glued to another word", which is the
    ///   space-or-punctuation boundary wanted here without demanding a literal space (so a
    ///   citation opening a line, or written "(2008) 1 SCC 1", still matches).
    /// - All-caps tokens also accept interior dots, so "A.I.R." matches alongside "AIR".
    /// - `(\s*\([A-Za-z&.\- ]{1,14}\))?` absorbs the court suffix in "AIR(SC)" / "CTR (Mad)".
    private static let citationRegex: NSRegularExpression = {
        func tolerant(_ token: String) -> String {
            let escaped = NSRegularExpression.escapedPattern(for: token)
            // "AIR" -> "A\.?I\.?R\.?" so the dotted spelling matches too. Only for all-caps
            // tokens; mixed-case ones ("AllMR", "KarLJ") are never printed with dots.
            guard token.allSatisfy({ $0.isUppercase || $0 == "&" }) else { return escaped }
            return token.map { "\(NSRegularExpression.escapedPattern(for: String($0)))\\.?" }.joined()
        }

        // Longest first so "AllMR" wins over a hypothetical shorter prefix.
        let tokens = (reporterTokens + neutralTokens)
            .sorted { ($0.count, $0) > ($1.count, $1) }
            .map(tolerant)
            .joined(separator: "|")

        var pattern = "(?<![A-Za-z])(?:\(tokens))(?:\\s*\\([A-Za-z&.\\- ]{1,14}\\))?(?![A-Za-z])"

        if includePartyMarker {
            // Minimal party marker — "v." / "vs." / "v/s" / "versus" between two words. Not a
            // party-name grammar, just a signal that a case is being named here.
            //
            // Case-INSENSITIVE and punctuation-tolerant on purpose: judgments write "Vs.", "VS."
            // and "Vs.," at least as often as "v.". Requiring lowercase and a following "\s\w"
            // (as this first did) lost roughly a third of the party-only references on a 116
            // judgment sample. The preceding "\w\s" still prevents a bare initial ("Mr. V.
            // Ramasamy") from matching, since "." is not a word character.
            // The bracketed separators also allow the hyphenated "-vs-" form, which is common
            // in Madras/Kerala formatting ("Masti Health ... -vs- Commissioner").
            pattern += "|(?<=\\w[\\s\\-])(?i:v|vs|v/s)\\.?(?=[\\s\\-,;:]|$)|(?i:\\bversus\\b)"
        }

        // A malformed pattern here is a programming error, not a runtime condition.
        return try! NSRegularExpression(pattern: pattern, options: [])
    }()

    // MARK: - Sentence boundaries

    /// Dots that do NOT end a sentence. Without this, a naive split shatters "A.I.R.",
    /// "D.B. Modak", "No. 12055" and "10.3.2005" into fragments, and the windows become
    /// meaningless. Deliberately short — it only has to be good enough to window text.
    private static let abbreviations = "No|Nos|Art|Sec|Cl|Ors|Anr|Ltd|Pvt|Co|Corp|Govt|Hon|Mr|Mrs|Ms|Dr|Prof|Rs|Vol|para|paras|pp|Ed|vs|v|etc|viz|Cri|Cr|LJ|JJ|AIR|ILR|SCC|SCR"

    /// UTF-16 offsets at which sentences start. Computed once per page.
    ///
    /// Masking replaces each non-terminal "." with a NUL, which preserves length exactly, so
    /// offsets found in the masked copy are valid in the original.
    private static func sentenceStarts(in text: NSString) -> [Int] {
        var masked = text as String

        func mask(_ pattern: String, _ template: String) {
            guard let re = try? NSRegularExpression(pattern: pattern) else { return }
            let full = NSRange(location: 0, length: (masked as NSString).length)
            masked = re.stringByReplacingMatches(in: masked, range: full, withTemplate: template)
        }

        mask("(?<=\\d)\\.(?=\\d)", "\u{0}")                    // 10.3.2005
        mask("(?<![A-Za-z])([A-Za-z])\\.", "$1\u{0}")           // A.I.R. / D.B.
        mask("\\b(\(abbreviations))\\.", "$1\u{0}")             // No. / Ltd. / vs.

        guard let re = try? NSRegularExpression(pattern: "(?<=[.!?])\\s+|\\n{2,}") else { return [0] }
        let full = NSRange(location: 0, length: (masked as NSString).length)
        var starts = [0]
        re.enumerateMatches(in: masked, range: full) { match, _, _ in
            if let end = match?.range.upperBound, end < text.length { starts.append(end) }
        }
        return starts
    }

    // MARK: - Public API

    /// True when the text contains anything citation-shaped. Cheap enough to run per page.
    static func containsCitation(_ text: String) -> Bool {
        let ns = text as NSString
        return citationRegex.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)) != nil
    }

    /// The passages worth sending to the model: every citation-shaped match plus `window`
    /// sentences either side, with overlapping windows merged.
    ///
    /// Returns `nil` when the text contains no candidate at all — the caller can skip that page
    /// entirely, which is where most of the saving comes from.
    static func candidatePassages(in text: String, window: Int = defaultSentenceWindow) -> String? {
        let ns = text as NSString
        let full = NSRange(location: 0, length: ns.length)
        let matches = citationRegex.matches(in: text, range: full)
        guard !matches.isEmpty else { return nil }

        let starts = sentenceStarts(in: ns)

        // Sentence index containing `offset` — binary search over the start offsets.
        func sentenceIndex(of offset: Int) -> Int {
            var low = 0, high = starts.count - 1, result = 0
            while low <= high {
                let mid = (low + high) / 2
                if starts[mid] <= offset { result = mid; low = mid + 1 } else { high = mid - 1 }
            }
            return result
        }

        var ranges: [(lo: Int, hi: Int)] = []
        for match in matches {
            let i = sentenceIndex(of: match.range.location)
            let lo = starts[max(0, i - window)]
            let hiIndex = i + window + 1
            let hi = hiIndex < starts.count ? starts[hiIndex] : ns.length
            // Never let the window cut through the match that produced it.
            ranges.append((lo: min(lo, match.range.location), hi: max(hi, match.range.upperBound)))
        }

        // Merge overlapping/adjacent windows so dense pages don't duplicate text.
        var merged: [(lo: Int, hi: Int)] = []
        for range in ranges.sorted(by: { $0.lo < $1.lo }) {
            if let last = merged.last, range.lo <= last.hi {
                merged[merged.count - 1].hi = max(last.hi, range.hi)
            } else {
                merged.append(range)
            }
        }

        let passages = merged.map {
            ns.substring(with: NSRange(location: $0.lo, length: $0.hi - $0.lo))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty }

        return passages.isEmpty ? nil : passages.joined(separator: "\n\n[…]\n\n")
    }
}
