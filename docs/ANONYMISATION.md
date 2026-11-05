# Anonymisation and data release policy

Status: draft of 2026-10-19, prepared for the authors. It is a technical and organisational plan, not legal advice. The open legal questions in Section 9 must be settled by the authors with the data protection officer (DPO) of their university before any personal-data-derived file is published.

## 1. Purpose and scope

This repository accompanies the paper on longitudinal privacy inference from documents that Italian public administrations publish under the transparency law (Legislative Decree 33/2013). The paper studies eight administrations, called Cases A to H, and promises to release "an anonymised version of the benchmark (inferred-attribute labels and detection corpora, with all direct identifiers and every reference to the administrations removed)", while the raw documents are not redistributed.

This file states which data may enter the repository, in which form and after which checks. It binds everyone who adds a file to the repository. Two rules hold without exception:

1. No name, acronym, web domain, URL, address, city or other reference that points to one of the eight administrations may appear anywhere in the repository: code, data, file names, folder names, figures, notebooks, commit messages or documentation. The cases are called A to H, with the type of body only (regional government, ministry, central agency, healthcare agency).
2. No information about a real person may appear in the public repository unless it has passed the transformation rules (Section 6) and the verification procedure (Section 7), and the release tier in Section 5 allows it.

Only data of the eight cases reported in the paper are in scope. Any other dataset found in the source material is out of scope and stays out of the repository.

## 2. Legal background in brief

### 2.1 Anonymous versus pseudonymous data under the GDPR

- The GDPR does not apply to anonymous information, that is, data that no longer relate to an identified or identifiable person, taking into account "all the means reasonably likely to be used" by the controller or by another person (GDPR, Recital 26) [1].
- Pseudonymised data (names replaced by codes, with the key kept separately; GDPR Art. 4(5)) are still personal data for anyone who holds or can reasonably obtain the key [1][3].
- The Article 29 Working Party's Opinion 05/2014 (WP216) assesses anonymisation against three risks: singling out, linkability and inference. It states that pseudonymisation is not anonymisation and describes generalisation (aggregation, k-anonymity, l-diversity, t-closeness) and randomisation (noise, permutation, differential privacy) [2].
- The EDPB Guidelines 01/2025 on pseudonymisation (adopted 16 January 2025) confirm that pseudonymised data remain personal data and introduce the "pseudonymisation domain": the people and systems that must not be able to re-identify the data subjects [3].
- In C-413/23 P, EDPS v SRB (4 September 2025), the Court of Justice held that pseudonymised data may be personal data for the controller and not for a recipient who has no reasonable means of re-identification [5].
- The EDPB draft Guidelines 02/2026 on anonymisation (version 1.0 adopted 7 July 2026, public consultation until 30 October 2026) update WP216. They turn the three risks into the criteria "no record isolation", "no linkage" and "no inference", follow the contextual approach of the SRB judgment, and treat inference through AI models as a "no inference" issue. They also warn that AI lowers the cost of re-identification, so anonymity must be reassessed over time [4]. The text is a draft and may change.

Consequences for this project:

- The inferred attributes are personal data about real, identifiable employees, whether or not the inferences are correct. Generated phishing messages that embed those attributes are personal data too.
- A file released to the public must be anonymous with respect to anyone in the world, including an attacker with a strong LLM and access to the same transparency portals the study used. That is the threat model of the paper itself.
- A file that keeps person-level pseudonyms and detailed career content is, at best, pseudonymised. It may be shared only under controlled access (tier 2 in Section 5), as personal data, with a legal basis, a data use agreement and the safeguards of GDPR Art. 89.

### 2.2 Reuse limits of data published under the transparency law

- Legislative Decree 33/2013, Art. 7 and Art. 7-bis: data subject to mandatory publication are public and reusable, but Art. 7-bis(1) allows reuse of personal data "nel rispetto dei principi sul trattamento dei dati personali" (in compliance with data protection principles). Art. 7-bis(4) obliges administrations to make non-pertinent personal data unintelligible, and Art. 7-bis(6) keeps the limits on data revealing health status [6].
- Publication is time-limited: as a rule five years (Art. 8(3)); data of office holders and managers stay online for three years after the end of the mandate (Art. 14(2)) [7][8]. A released dataset must not keep alive, in a new place, information that the law requires to be taken down.
- The Garante's 2014 guidelines on transparency publications (doc. web n. 3134436, Part I §6) state that the online availability of personal data does not make them freely reusable. Reuse must be compatible with the transparency purpose; it may be limited to anonymised and aggregated data; open licences without data protection clauses are not enough, and licences should forbid re-identification. Sensitive and judicial data are excluded from reuse [8].

The study reused the data for research. Republishing them, even in derived form, is a new act of dissemination that needs its own justification. This is why the plan below is conservative.

### 2.3 Research provisions

GDPR Art. 89 requires safeguards for processing for scientific research purposes, pseudonymisation and minimisation among them. In Italy, Art. 110-bis of the Personal Data Protection Code governs further processing for research, and the Garante's "Regole deontologiche" for statistical and scientific research (provv. n. 515 of 19 December 2018, Annex A5 to the Code) apply; new rules were being prepared in 2025 [9]. The EDPB adopted draft guidelines on processing for scientific research in April 2026 [10]. Which of these provisions applies to this study is an open question (Section 9).

## 3. Methods for text data: what works and what does not

**Pattern-based detection.** Regular expressions find structured identifiers reliably: e-mail addresses, Italian phone numbers, tax codes (codice fiscale, with checksum), VAT numbers, IBANs, URLs and domains. `scripts/anonymisation_check.py` implements them.

**Named-entity recognition (NER).** The spaCy Italian pipelines `it_core_news_lg` and `it_core_news_sm` include a named-entity recogniser for persons, organisations and locations [11]. Microsoft Presidio adds context-aware recognisers, including Italy-specific ones (IT_FISCAL_CODE, IT_VAT_CODE, IT_DRIVER_LICENSE, IT_PASSPORT, IT_IDENTITY_CARD) with pattern, context and checksum logic [12]. NER misses names written in capitals, inverted ("SURNAME NAME"), joined by underscores, or embedded in file names, which is exactly how this project's data store them. NER is therefore a second line of defence. The first line is exact matching against the list of known names taken from the source data (Section 6.1).

**Generalisation of quasi-identifiers.** Role, organisational unit, dates and career events identify a manager in a given administration even when the name is gone: an attacker who knows the case type can search the transparency portals. Generalisation replaces exact values with coarser classes (exact dates with periods, job titles with role categories, units with functional labels) [2][14]. It is the main protection for this project, because the dangerous content is the career profile, not only the name.

**Surrogate replacement.** Detected identifiers can be replaced with realistic fake values used consistently within a document. Realistic surrogates hide the identifiers that the detector missed, because a leaked real name looks like one more fake name ("hiding in plain sight", Carrell et al. 2013) [13]. The same property makes residual leaks harder to spot in review, so surrogates must come from a closed, documented list that the scanner can exclude, and the review must compare against the known-names list.

**Placeholders.** Tags such as `[PERSON]` are transparent and easy to audit, but they change the text distribution. In a phishing-detection corpus a classifier can learn the tags instead of the phishing cues. If placeholders are used, both classes must receive the same treatment.

**LLM-assisted rewriting.** Language models can rewrite text to remove inferable personal information, and adversarial LLM anonymisation outperforms commercial tools in the evaluation of Staab et al. [15]. The risks for this project: (a) sending personal data to a cloud API is itself a processing operation and a transfer, so only local models are acceptable; (b) rewriting changes content, so the judge labels attached to an attribute may no longer hold; (c) output is not deterministic and may hallucinate or drop identifiers silently. An LLM pass may complement the deterministic rules, never replace them.

**Limits of text sanitisation.** Surface-level sanitisation gives a false sense of privacy: Xin et al. re-identified individuals in sanitised datasets by linking residual, innocuous-looking content to auxiliary information [16]. Word-level differential privacy leaves contextual clues that LLMs can exploit to reconstruct the original text (Meisenbacher et al., WPES 2025) [17]; Pang et al. report high recovery rates against word-level DP sanitisation [18]. Lison et al. and the Text Anonymization Benchmark argue for measuring disclosure risk explicitly instead of trusting entity tagging [19][20]. The paper's own results show that LLMs reconstruct implicit attributes from administrative text. Anything released must therefore be tested against an LLM attacker (Section 7.3).

**Residual-risk testing.** A "motivated intruder" test, in which a competent person with web access tries to re-identify records, is expected by the UK ICO for any claim of anonymity [21]. NIST SP 800-188 recommends re-identification studies, a review board and a choice of data-sharing model (public de-identified data, synthetic data, query interface, protected enclave) [14].

**Release tiers.** Data can be public, available on request under conditions, or kept in-house. On Zenodo, files can be public or restricted while the record metadata stay public; restricted files can be shared with approved users, and embargoes are possible [22]. The access conditions (for example a signed data use agreement) must be written by the authors; Zenodo does not provide a standard agreement.

## 4. Data inventory of the source material

Counts only, from a read-only scan of the source material on 2026-10-19. "Persons" means profiles; Case G includes organisational entities as well as people. Direct identifiers are counted with the patterns of `scripts/anonymisation_check.py`, so the figures are lower bounds for names, which also appear in forms the patterns cannot catch.

<!-- INVENTORY_TABLE -->

## 5. Release tiers

| Tier | Where | Who | Condition |
|---|---|---|---|
| 0, public now | this repository and its Zenodo snapshot | anyone | contains no person-level content |
| 1, public after transformation | this repository, later version | anyone | Section 6 rules applied, Section 7 checks passed, authors' and DPO's written approval |
| 2, controlled access | restricted Zenodo record or institutional repository | approved researchers who sign a data use agreement | pseudonymised personal data; legal basis confirmed; agreement in place |
| 3, never released | authors' access-controlled storage | the authors | retention period set by the authors |

Plan per data type:

| Data type | Tier | Form |
|---|---|---|
| Pipeline code | 0 | after removing portal URLs, base URLs, administration names, crawler targets specific to a portal, API keys, personal paths and contributor names |
| Prompt templates | 0 | generic templates only, with no example drawn from a real document |
| Aggregate metrics per case and per exposure step (attribute counts, DIR, hallucination rate, explicitness, PES curve, IQS) | 0 | case letter only; no per-person rows |
| Detection results per model and per corpus (accuracy, precision, recall, F1) | 0 | numbers only; no per-message predictions |
| Figures | 0 | aggregate figures only, checked by eye for labels, legends and example text |
| Judge labels without text | 0 | one row per attribute: case, random person ID, step, attribute index, the four majority labels, vote counts, number of judges that saw the attribute. No attribute text, no judge notes. Case F (18 profiles) only as per-step aggregates |
| Data dictionary and this policy | 0 | |
| Generated phishing e-mails (BEC and spear phishing) | 1 | after Section 6.4; Cases A, B, C, D, E and H only; Cases F and G stay in tier 2 (small cohort; organisational entities and collective-bargaining content) |
| Legitimate Italian e-mails (negative class) | 1 or 3 | tier 1 only if their origin and licence allow redistribution and they pass the same scan; otherwise give retrieval instructions, not the texts |
| Detection results recomputed on the released corpus | 1 | to be released with the corpus, because the transformed corpus is not the corpus used in the paper |
| Inferred-attribute texts | 2 | pseudonymised and redacted (Section 6.3); no public release of free-text attributes |
| Judge notes (free-text rationales) | 2 | same rules as the attribute texts |
| Per-person metrics (PES, DIR, HR per profile) | 2 | random person IDs, no document counts |
| Public phishing benchmarks (CEAS_08, Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin) | 3 (not redistributed) | third-party data with real senders and recipients; give the source, the version and checksums, and release the filtering code. The compiled Kaggle dataset asks users to cite Al-Subaiey et al. (arXiv 2405.11619) [23]; check its licence before linking any derived file |
| Raw documents (PDF, HTML, spreadsheets) and extracted texts | 3 | |
| Crawled personnel tables (names, phone numbers, e-mail addresses, compensation, dates in office) | 3 | |
| Per-person LLM caches, threat-simulation texts, prompts and raw model responses | 3 | they contain document text and attack plans aimed at named people |
| Mapping between pseudonyms and real names, known-names list, forbidden-terms list | 3 | stored encrypted, apart from the data, by one named author |
| Trained classifiers and fine-tuned checkpoints (pickles, Transformer weights, TF-IDF vocabularies) | 3 | models trained on personalised messages can memorise names and units |
| Student reports, notebooks with outputs, `.DS_Store` files, archives | 3 | they contain administration names, file names and examples |
| API keys | 3 | must also be revoked (several were found in the source code) |

## 6. Transformation rules

All transformations run on the authors' machines, with local tools only. The transformation code may be published; the lexicons that contain real names or units (known-names list, unit lexicon, forbidden terms) may not.

### 6.1 Inputs that the authors build first (kept in tier 3)

1. `known_names.txt`: every person name in the source material, one per line, taken from the per-person keys of the result files, the per-person folder and file names, and the crawled personnel tables. Include colleagues and signatories found by NER in the documents.
2. `forbidden_terms.txt`: for each administration its full name, acronym, web domains, portal names, the city, province and region where it is based, the names of its departments and units, the programmes, projects and funds it manages (programme codes included), regional laws, and any distinctive building or street name. Add the generic forms that identify a body only in combination (for example "Agenzia" followed by a sector name).
3. `unit_lexicon.csv`: each organisational unit that occurs in the data, mapped to one functional category of Section 6.3.
4. `role_lexicon.csv`: each job title, mapped to one role category of Section 6.3.

### 6.2 Identifiers and pseudonyms

- Person IDs: `Case<X>-P<NNN>`, for example `CaseB-P017`, assigned by a random permutation within each case (not alphabetical, not in crawl order). Organisational entities in Case G: `CaseG-O<NNN>`.
- Attribute IDs: `<person ID>-S<step>-A<NN>`, for example `CaseB-P017-S3-A04`.
- Message IDs: `Case<X>-E<NNNN>`, assigned by a separate random permutation. In tier 1, messages carry no person ID, so a reader cannot gather the messages and labels of one person.
- The random seed and the mapping table stay in tier 3. Delete them when the retention period agreed with the DPO ends; once they are gone, the authors can no longer link the released data to people.
- File and folder names in the release use IDs only. No file name may be derived from a person's name, not even in hashed form (a hash of a name can be recomputed from the public portal).

### 6.3 Fields and quasi-identifiers

Drop in every tier: names; e-mail addresses; phone numbers; tax codes; addresses; IBANs; compensation amounts; the document file names; timestamps of crawling and inference; source URLs; the raw model response; the prompt; the number of documents per person.

Generalise as follows (tier 1 always; tier 2 for units, locations and named people, with the rest kept as is):

| Content | Rule |
|---|---|
| Person names (the subject and anyone else) | Tier 1: surrogate from a fixed list of common Italian first names and surnames, consistent within a message, gender kept for grammatical agreement, and never equal to an entry of `known_names.txt`. Tier 2: `[PERSON]`, `[PERSON_2]`, ... |
| Administration | Tier 1: one fictitious name per case drawn from a published list (for example "Ente Alfa" for a central agency), never a real body. Tier 2: `[ADMINISTRATION]` |
| Organisational unit | functional category: general management, finance and budget, human resources, legal affairs, procurement, IT and digital services, inspection and control, planning and statistics, communication, international affairs, health services, technical services, other |
| Job title | role category: head of administration; general manager (dirigente di prima fascia or equivalent); manager (dirigente); organisational position or high professional role (PO/EQ); official (funzionario); medical manager; health professional; administrative staff; other. A title that occurs fewer than 5 times in a case becomes the next broader category |
| Dates | exact dates and months removed; years in tier 1 replaced by a per-message random shift of 1 to 3 years that keeps the order of events; durations rounded to 5-year bands ("for more than ten years") |
| Places (city, province, region, address, building) | `[LOCATION]` or a surrogate from a fixed list of Italian cities that does not include the city of the case |
| Education | field of study category (law, economics, engineering, medicine, humanities, sciences, other); universities and schools removed |
| Programmes, projects, funds, regional or internal regulations, protocol numbers, decree numbers | `[PROGRAMME]`, `[REGULATION]`, `[ID]`. National laws of general application (for example Legislative Decree 33/2013) may stay |
| Amounts of money | removed in tier 1; bands of EUR 25,000 in tier 2 |
| Health, union membership, political opinions, religion, criminal matters (special categories, GDPR Art. 9 and 10) | the whole attribute or sentence is removed in every tier |
| Rare events (awards, publications, named committees, a unique career move) | removed in tier 1; generalised in tier 2 |

### 6.4 Generated phishing e-mails (tier 1)

1. Parse each message into subject, salutation, body and signature.
2. Apply Sections 6.2 and 6.3 to every field. The sender's address and display name become surrogates under the reserved domain `.example` (RFC 2606), for example `segreteria@ente-alfa.example`, so no real domain can appear.
3. Links in the body: replace the host with a `.example` host and keep the path shape, because URL features matter for detection.
4. Apply the identical procedure, with the same surrogate lists, to the legitimate e-mails that form the negative class, so the two classes differ only in what they say.
5. Retrain and re-evaluate the detectors on the transformed corpus and release those numbers next to the original ones, stating that the paper's figures were computed on the untransformed corpus.

### 6.5 Inferred attributes and judge notes (tier 2)

The text of an inferred attribute is often a sentence such as "<name>'s department: ... which suggests that ...". Replace the subject's name with "The person", apply the tier 2 rules of Section 6.3, and remove any attribute that falls in a special category. Keep the four labels and the votes unchanged. If the transformation removes the content of an attribute (for example an attribute that only states an e-mail address), keep the row with the text `[REMOVED: direct identifier]` so that counts and metrics remain reproducible.

### 6.6 Order of operations

1. Exact replacement of known names and forbidden terms (longest match first; tokens are compared case-insensitively, without accents, and with spaces, underscores, hyphens and apostrophes treated as equal).
2. Regular expressions for structured identifiers.
3. Lexicon-based generalisation of units, roles, places and dates.
4. NER (`it_core_news_lg`) for remaining persons, organisations and places, with every hit reviewed.
5. Optional local LLM pass that flags, and does not rewrite, sentences from which a reader could infer the administration or the person.
6. Scan and review (Section 7).

## 7. Verification procedure

### 7.1 Automatic scan

Run on every candidate file and on the whole repository before each commit that adds data:

```
python scripts/anonymisation_check.py <paths> \
    --forbidden-file <private>/forbidden_terms.txt \
    --names-file <private>/known_names.txt --ner \
    --allow-file docs/anonymisation_allow.txt
```

The run must end with `RESULT: PASS`. Every NER and image finding must be looked at. A finding is cleared only by fixing the data or by adding a narrow pattern to a local allow file with a comment that says why.

### 7.2 Manual review of a random sample

- For each tier 1 file type, draw a random sample of 300 items, stratified by case (at least 30 items per case, or all items when a case has fewer). Record the seed and the list of sampled IDs.
- Two reviewers who know the administrations read the sample independently, against the known-names list and the forbidden-terms list. They flag direct identifiers, references to an administration, and combinations of details that would let them find the person on a portal.
- Acceptance: zero flagged items. With zero failures in 300 items, the 95% upper bound of the residual failure rate is about 1% ("rule of three", Hanley and Lippman-Hand 1983) [24]. For a bound of 0.5%, review 600 items.
- If any item fails, correct the rule (not only the item), rerun the whole transformation, and draw a new sample.
- Tier 2 files: same procedure with 150 items per file type (95% bound about 2%).

### 7.3 Re-identification test

- Motivated intruder: two people who did not do the transformation each receive 25 sampled tier 1 items and 30 minutes per item, with web search and the transparency portals. They try to name the administration and the person.
- LLM attacker: a local model (for example the inference model of the paper, run on the authors' machine) receives 100 sampled items and is asked to name the administration, the region and the person, with its confidence.
- Acceptance: no person correctly identified. Any correct identification of an administration is analysed, the rule that let it through is extended, and the test is repeated on a new sample.

### 7.4 Sign-off

Keep in tier 3 a short record per released file: version of the transformation code, scan output, sample seed, reviewers, intruder and LLM test results, date, and the authors' and DPO's approval. Repeat the scan and the LLM test before each new public version, since attack capabilities change over time [4].

## 8. Checklist before adding any file to the repository

1. The file is listed in Section 5 with tier 0, or with tier 1 and a completed sign-off record.
2. The file comes from one of Cases A to H, and its name and path contain no administration, person, contributor or portal name.
3. It contains no raw document, extracted text, prompt, model response, threat simulation, crawled table, trained model, notebook output or archive.
4. The code contains no portal URL, base URL, administration name, absolute personal path or API key; configuration values are placeholders or environment variables.
5. Tables contain no column with names, e-mail addresses, phone numbers, compensation, document file names or document counts per person.
6. Judge labels are released without attribute text and without judge notes.
7. The automatic scan of Section 7.1 returns `RESULT: PASS` on the file and on the whole repository.
8. Every figure has been opened and checked for names, units, places and example text.
9. Third-party benchmark data are referenced, not copied.
10. `forbidden_terms.txt`, `known_names.txt`, allow files, mapping tables and scan outputs produced with `--reveal` are outside the repository and listed in `.gitignore`.
11. `git log -p` and the commit messages contain none of the above either; history that ever contained personal data must be rewritten before the repository goes public.
12. When in doubt, the file stays out and the question goes to the authors.

## 9. Open questions for the authors

1. Legal basis for republishing derived data: public interest task (GDPR Art. 6(1)(e) with Art. 2-ter of the Italian Code) or legitimate interest (Art. 6(1)(f)), and whether Art. 110-bis of the Code or the research rules of conduct apply. The ethics statement of the paper names both bases with a note to verify.
2. Whether a data protection impact assessment is needed: the study infers attributes about identifiable employees on a large scale and generates attack messages about them.
3. Information to data subjects: whether the exemption of GDPR Art. 14(5)(b) (disproportionate effort, research) is relied upon, and whether a notice is published with the dataset.
4. Compatibility of the reuse with the transparency purpose (Garante guidelines 2014, Part I §6) and with the publication periods of Art. 8 and Art. 14(2) of Legislative Decree 33/2013, for data that portals may have removed since the crawl.
5. Whether tier 2 is needed at all, who approves access, which data use agreement is used (no re-identification, no contact with data subjects, no onward sharing, deletion after the project, report of any accidental re-identification), and the retention period of tier 3 data and of the mapping key.
6. Origin and licence of the legitimate Italian e-mails used as the negative class, and whether they contain personal data of third parties.
7. Whether the generated phishing corpus should be public at all, given its dual use, or kept in tier 2 with the attribute texts.
8. Revocation of the API keys found in the source code, and removal of the source history from any place where it may have been shared.

## 10. References

All sources accessed on 2026-10-19.

1. Regulation (EU) 2016/679 (GDPR), Recital 26, Art. 4(5), Art. 9, Art. 14(5)(b), Art. 89. https://eur-lex.europa.eu/eli/reg/2016/679/oj
2. Article 29 Data Protection Working Party, Opinion 05/2014 on Anonymisation Techniques, WP216, 10 April 2014. https://ec.europa.eu/justice/article-29/documentation/opinion-recommendation/files/2014/wp216_en.pdf
3. EDPB, Guidelines 01/2025 on Pseudonymisation, version for public consultation, adopted 16 January 2025. https://www.edpb.europa.eu/our-work-tools/documents/public-consultations/2025/guidelines-012025-pseudonymisation_en ; summary: https://www.hunton.com/insights/publications/edpb-advises-on-pseudonymisation-for-gdpr-compliance
4. EDPB, Guidelines 02/2026 on Anonymisation, version 1.0 adopted 7 July 2026, public consultation until 30 October 2026. https://www.edpb.europa.eu/public-consultations/guidelines-022026-on-anonymisation_en ; PDF: https://www.edpb.europa.eu/system/files/2026-07/edpb_guidelines_202602_anonymisation_v1_en_0.pdf
5. Court of Justice of the EU, C-413/23 P, EDPS v SRB, judgment of 4 September 2025. https://curia.europa.eu/juris/liste.jsf?num=C-413/23 ; commentary: https://www.cliffordchance.com/insights/resources/blogs/talking-tech/en/articles/2025/09/pseudonymized-data-after-edps-v-srb.html
6. Legislative Decree 14 March 2013, n. 33, Art. 7 and Art. 7-bis (text as consolidated). https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legislativo:2013-03-14;33 ; Art. 7-bis: https://www.codiceappalti.it/DLGS_33_2013/Art__7-bis__Riutilizzo_dei_dati_pubblicati/9896
7. ANAC, on publication periods under Art. 8(3) and Art. 14(2) of Legislative Decree 33/2013. https://www.anticorruzione.it/documents/91439/376b37f6-6608-0242-65fb-fd571c465527
8. Garante per la protezione dei dati personali, "Linee guida in materia di trattamento di dati personali, contenuti anche in atti e documenti amministrativi, effettuato per finalità di pubblicità e trasparenza sul web da soggetti pubblici e da altri enti obbligati", 15 May 2014, doc. web n. 3134436 (Part I §6 reuse, §7 duration, §9.a curricula, §9.c compensation). https://www.garanteprivacy.it/garante/doc.jsp?ID=3134436
9. Italian Personal Data Protection Code (Legislative Decree 196/2003), Art. 2-ter and Art. 110-bis; Garante, Regole deontologiche per trattamenti a fini statistici o di ricerca scientifica, provv. n. 515 of 19 December 2018 (Annex A5). https://www.garanteprivacy.it/home/docweb/-/docweb-display/docweb/9069637 ; on new rules in preparation: https://www.agendadigitale.eu/sicurezza/privacy/ricerca-e-protezione-dati-verso-nuove-regole-deontologiche-un-passo-cruciale/
10. EDPB, news of 16 April 2026 on the guidelines on processing of personal data for scientific research purposes. https://www.edpb.europa.eu/news/edpb-brings-clarity-to-data-processing-for-scientific-research-speeds-up-the-finalisation-of_en
11. spaCy, Italian trained pipelines (it_core_news_sm, it_core_news_lg). https://spacy.io/models/it
12. Microsoft Presidio, Supported entities (Italy: IT_FISCAL_CODE, IT_DRIVER_LICENSE, IT_VAT_CODE, IT_PASSPORT, IT_IDENTITY_CARD). https://presidio.dataprivacystack.org/supported_entities/
13. D. Carrell et al., "Hiding in plain sight: use of realistic surrogates to reduce exposure of protected health information in clinical text", JAMIA 20(2), 2013. https://academic.oup.com/jamia/article/20/2/342/897812
14. NIST SP 800-188, De-Identifying Government Datasets: Techniques and Governance, September 2023. https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-188.pdf
15. R. Staab, M. Vero, M. Balunović, M. Vechev, "Language Models are Advanced Anonymizers", ICLR 2025 (arXiv 2402.13846). https://arxiv.org/abs/2402.13846
16. R. Xin, N. Mireshghallah et al., "A False Sense of Privacy: Evaluating Textual Data Sanitization Beyond Surface-level Privacy Leakage", arXiv 2504.21035, 2025. https://arxiv.org/abs/2504.21035
17. S. Meisenbacher, A. Klymenko, A.-E. Bodea, F. Matthes, "The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization", WPES @ CCS 2025 (arXiv 2508.18976). https://arxiv.org/abs/2508.18976
18. S. Pang, Z. Lu, H. Wang, P. Fu, Y. Zhou, M. Xue, "Reconstruction of Differentially Private Text Sanitization via Large Language Models", RAID 2025 (arXiv 2410.12443). https://arxiv.org/abs/2410.12443
19. P. Lison, I. Pilán, D. Sánchez, M. Batet, L. Øvrelid, "Anonymisation Models for Text Data: State of the art, Challenges and Future Directions", ACL-IJCNLP 2021, pp. 4188-4203. https://aclanthology.org/2021.acl-long.323
20. I. Pilán, P. Lison, L. Øvrelid, A. Papadopoulou, D. Sánchez, M. Batet, "The Text Anonymization Benchmark (TAB)", Computational Linguistics 48(4), 2022, pp. 1053-1101. https://aclanthology.org/2022.cl-4.19
21. UK Information Commissioner's Office, Anonymisation guidance, "How do we ensure anonymisation is effective?" (motivated intruder test). https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-sharing/anonymisation/how-do-we-ensure-anonymisation-is-effective/
22. Zenodo documentation, file visibility (public or restricted, embargo; metadata always public). https://help.zenodo.org/docs/deposit/create-new-upload/ and https://help.zenodo.org/docs/deposit/manage-files/
23. Phishing Email Dataset (compilation of CEAS_08, Enron, Ling, Nazario, Nigerian Fraud, SpamAssassin), Kaggle, credited to N. A. Alam; citation requested: Al-Subaiey et al., arXiv 2405.11619. https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset ; https://arxiv.org/abs/2405.11619
24. J. A. Hanley, A. Lippman-Hand, "If nothing goes wrong, is everything all right? Interpreting zero numerators", JAMA 249(13):1743-1745, 1983. doi:10.1001/jama.1983.03330370053031
