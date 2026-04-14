# Technical Report: 

## 1. Context

In recent years, concerns have emerged over declining employment outcomes among fresh graduates in Singapore. The 2025 Graduate Employment Survey (GES) reported a decline in full-time permanent employment despite stable median salaries (The Straits Times, 2025), while surveys indicate increasing anxiety among graduates on job prospects (Channel NewsAsia, 2025).

These trends raise important questions about the alignment between higher education and labour market demands. Current analyses rely on aggregate outcomes, lacking visibility into specific skills or curriculum components contributing to employability outcomes. While universities continue to equip students with theoretical knowledge and foundational skills, the evolving nature of industry requirements, driven by technological advancements and shifting economic conditions, may result in a mismatch between what is taught and what employers seek.

Hence, this project answers a key question: How well are university courses preparing students for real-world jobs? By analysing job descriptions alongside university course content, we aims to systematically evaluate the extent to which academic curricula align with current industry skill requirements, and to identify potential gaps that may contribute to graduate employment challenges.

## 2. Scope

### 2.1 Problem

#### Problem definition 
MOE’s Higher Education Policy Division (HEPD) faces the ongoing challenge of ensuring that university curricula remain aligned with labour market demands. As the body responsible for higher education policy and quality assurance, HEPD must regularly assess whether graduates possess the skills required by employers.
However, this task is inherently complex due to the nature of the data involved. Job advertisements and course descriptions are large-scale, unstructured, and continuously evolving sources of information. Thousands of job postings are generated regularly, with skills described using varied, inconsistent, and context-dependent language. Similarly, course descriptions differ across institutions in structure, terminology, and level of detail.
This problem occurs continuously as labour market demands shift rapidly due to technological advancements and industry changes. In the absence of an automated system, HEPD relies on manual reviews or periodic audits conducted over multi-year cycles. These approaches are resource-intensive and unable to keep pace with real-time changes, increasing the risk that curriculum evaluations are based on outdated or incomplete information.

#### Impact and Significance 
The lack of a scalable and systematic approach to assessing curriculum–labour market alignment has several key consequences:
Graduate employability challenges: The 2025 Graduate Employment Survey (GES) reports a decline in full-time employment among fresh graduates, suggesting potential mismatches between acquired skills and employer expectations (The Straits Times, 2025).
Increased job search anxiety: Graduates report heightened uncertainty and stress in securing employment, reflecting concerns about their preparedness for the job market (Channel NewsAsia, 2025).
Inefficient curriculum planning: Universities may continue offering courses that are not closely aligned with industry needs, leading to suboptimal allocation of educational resources.
Delayed policy response: Without timely insights, MOE may take years to identify emerging skill gaps, limiting its ability to respond proactively.
Scalability limitations: Manually analysing thousands of job postings and course descriptions is impractical, making continuous monitoring infeasible.
Collectively, these issues weaken the ability of the higher education system to produce graduates who are well-prepared for an evolving workforce.

#### Why Data Science / Machine Learning is Appropriate
Data science and machine learning provide a suitable solution due to their ability to process large-scale, unstructured, and dynamic data.
First, Natural Language Processing (NLP) techniques enable the extraction and standardisation of skill-related information from both job advertisements and course descriptions, allowing meaningful comparison despite differences in wording.
Second, embedding-based models can represent both datasets within a shared semantic space, where similarity measures (e.g., cosine similarity) quantify the alignment between courses and job requirements. This transforms qualitative text into measurable indicators.
Finally, automated data pipelines enable continuous and scalable analysis, allowing MOE to monitor labour market trends in near real-time rather than relying on infrequent manual reviews. This improves both the speed and reliability of insights, supporting more responsive and data-driven policy decisions.
Overall, data science and machine learning directly address the challenges of scale, variability, and timeliness, making them well-suited for evaluating curriculum relevance in a rapidly changing labour market.



### 2.2 Success Criteria

#### Business Goals
If our project is successful, it can improve alignment between university curricula and labour market demand. The project enables stakeholders (e.g., MOE, universities) to identify gaps between skills taught in courses and skills required in job advertisements. Success is achieved if the system can consistently highlight high- and low-alignment courses, supporting data-driven curriculum improvements.

Another success outcome is enhancing graduate employability insights. By linking courses to relevant job opportunities and associated salary signals, the system provides actionable insights into which courses are most aligned with industry needs. Success is reflected in the ability to generate meaningful rankings or recommendations that inform students, educators, and policymakers.

#### Operational Goals 
Scalable and efficient processing of large unstructured datasets is another success outcome. The system should be able to process thousands of job postings and course descriptions efficiently using automated pipelines (e.g., text cleaning, skill extraction, embeddings). Success is achieved if the pipeline runs reliably within a reasonable time (e.g., minutes instead of manual analysis).

Accurate and consistent skill extraction and matching should also be achieved. The system should produce reliable representations of skills from both job and course data. Success is indicated by consistent alignment scores that reflect meaningful overlaps between courses and jobs, avoiding noise from irrelevant or low-quality skill data.

### 2.3 Assumptions

This project is based on several key assumptions that influence its feasibility and the validity of its outputs.

First, it is assumed that job advertisements provide a reliable proxy for labour market demand. While job postings reflect employer requirements, they may not always fully capture actual job responsibilities or may include inflated or generic skill requirements. 
Second, the analysis assumes that university course descriptions accurately represent the skills and knowledge acquired by students. In practice, actual learning outcomes may vary depending on teaching methods, assessments, and informal learning experiences.

Third, the project assumes that NLP techniques can effectively extract meaningful skill signals from unstructured text. However, some skills may be implicit, context-dependent, or described inconsistently, which could affect extraction accuracy.

Fourth, it is assumed that semantic similarity between course content and job descriptions is a valid indicator of real-world alignment. While embedding models capture textual similarity, they may not fully reflect the depth or practical applicability of skills.

Finally, it is assumed that sufficient and representative data is available, and that stakeholders (e.g., MOE and universities) are able to act on the insights generated. Without adequate data quality or institutional adoption, the system’s impact would be limited.


## 3. Overall Report Structure

This report is organised into three main analytical parts:

- **Part I: Data Cleaning and Preparation**
  - documents how raw job and course data were collected, cleaned, normalised, and converted into reusable analysis-ready datasets
- **Part II: General Pipeline**
  - explains the main alignment pipeline used to answer the project question
  - compares the baseline and experimental approaches and justifies why the baseline pipeline is the main reporting approach
- **Part III: STEM-Focused Pipeline**
  - explains why a STEM-only pipeline was explored
  - motivates STEM scoping as a way to reduce noise in the broader dataset
  - compares STEM findings against the general pipeline findings

This structure reflects the final project logic: the general pipeline provides the main answer to the project question, while the STEM pipeline acts as a narrower sensitivity analysis.

## 4. Methodology

### Part I: Data Cleaning and Preparation

### 3.1 Notebook 

Methodologically, the notebook follows a standard data engineering pattern:

1. Ingest raw semi-structured JSON records.
2. Standardise nested fields into tabular columns.
3. Filter for the target population of interest.
4. Engineer interpretable features.
5. Clean and regularise skills.
6. Persist analysis-ready outputs.
7. Run descriptive analyses to validate whether the cleaned dataset reflects plausible labour-market patterns.

This sequence separates structural cleaning from analytical interpretation while keeping both in the same artifact for transparency.

### 3.1A Project-Wide Pipeline Overview

- Add a short end-to-end overview of the full project workflow:
  - raw job and module data acquisition
  - scraper outputs
  - notebook-based cleaning
  - downstream dataset construction
  - canonical skill mapping
  - module-job alignment
- Clarify that the report should ultimately cover the full project system, not only the jobs notebook.
- State that the notebook-cleaned PKL files are now the source of truth for downstream workflows.
- Mention the standardized shell-script entrypoints for the supported pipelines.

### 3.2 Data Collection and Ingestion

The loader searches `../../data` recursively for files whose names begin with `MCF-`, falling back to a `job` subdirectory only if needed. This is a robust engineering decision because it prioritises the intended project data while remaining resilient to folder reorganisation. During execution, the notebook discovered **22,718 raw JSON files** and loaded **22,718 job rows**.

Each record is flattened into a structured row with fields such as:

- `uuid`
- `title`
- `description`
- `minimum_years_experience`
- `skills`
- `employment_types`
- `position_levels`
- `categories`
- `salary_minimum`, `salary_maximum`
- posting and expiry dates
- `ssoc_code` and `ssoc_version`

The notebook also strips HTML from descriptions using `BeautifulSoup`, which is important because job descriptions are often stored as HTML fragments rather than plain text. This reduces noise before text-based filtering and makes length checks more meaningful.

### 3.3 Targeted Cleaning for Graduate-Relevant Roles

The first major cleaning stage aligns the dataset to the project objective: identifying labour demand relevant to undergraduates and recent graduates.

The pipeline applies the following filters:

- Keep only postings with `minimum_years_experience` in `{0, 1}`.
- Drop rows missing title or description.
- Remove descriptions with fewer than 10 words.
- Remove internships using title, description, and employment-type signals.
- Remove likely postgraduate roles using title cues such as "research fellow" or "assistant professor".
- Remove postings whose descriptions strongly indicate postgraduate qualifications, including PhD or Master's requirements.
- Deduplicate records using the pair `(title, description)`.

The observed row counts show the effect of each stage:

| Stage | Rows Remaining |
|---|---:|
| Raw loaded postings | 22,718 |
| After experience filter | 9,477 |
| After description filter | 9,476 |
| After undergraduate-only filter | 8,834 |
| After deduplication | 7,115 |
| After skill thresholding | 7,104 |

These filters demonstrate robustness in two ways. First, they address known data quality issues such as sparsity and duplication. Second, they encode domain logic rather than relying on generic preprocessing. In a public-sector context, that matters because the distinction between internship, graduate, and postgraduate pipelines is policy-relevant: interventions for undergraduate curriculum design should not be distorted by jobs intended for researchers or late-stage professionals.

### 3.4 Employment Type, Salary, and Imputation Logic

The notebook derives `contract_type` and `work_type` from employer-provided `employment_types`, mapping values into interpretable categories such as `Permanent`, `Contract`, `Temporary`, `Freelance`, `Full Time`, and `Part Time`.

If both contract type and work type are unknown, the row is removed. This avoids carrying forward records with insufficient labour-market signal.

The notebook then converts salary fields to numeric format and computes `avg_salary` as the rounded mean of minimum and maximum salary. For work type, the notebook implements a two-step imputation strategy:

- First, infer likely work type using the modal observed work type within the same 3-digit SSOC group.
- If SSOC-based inference is unavailable, fall back to a salary threshold derived from the median salaries of known full-time and part-time jobs.

It compromises between practical utility and interpretability. It is more principled than filling all missing work types with the dominant class, because it uses occupational structure first and only uses salary as a weaker fallback. In public-sector analytics, such hierarchy-based imputation is preferable because it better preserves real labour-market structure.

After cleaning, the final job dataset contains:

- **7,104 rows**
- **6,448 full-time postings**
- **656 part-time postings**
- Contract types dominated by `Unknown` (3,133) and `Permanent` (2,781), followed by `Contract` (804), `Temporary` (331), and `Freelance` (55)

The large `Unknown` contract-type share is itself an important analytical finding: it reflects incomplete source metadata and should be acknowledged in any downstream interpretation.

### 3.5 Skill Normalisation and Frequency Filtering

The skill cleaning process has several stages:

1. Lowercase and trim raw skills.
2. Save raw skill frequencies to Excel for auditability.
3. Normalise punctuation and spacing.
4. Remove explicitly low-value labels such as `team player`, `able to work independently`, and `physically fit`.
5. Collapse variants of common soft skills into shared canonical forms, such as mapping phrases containing "communication" to `communication`.
6. Protect selected exact multi-word skills such as `project management` and `data management` from over-collapsing.
7. Remove within-row near-duplicates using fuzzy matching (`SequenceMatcher`).
8. Keep only skills that appear at least three times across the dataset.
9. Remove jobs with fewer than three cleaned skills.

This design balances precision and recall. If the notebook kept every raw employer phrase, the analysis would be overwhelmed by lexical variation and boilerplate. If it over-normalised aggressively, it would erase meaningful distinctions between technical competencies. The use of exact-keep exceptions and fuzzy deduplication shows good practical understanding of this trade-off.

The final distribution of skill counts is plausible for job postings:

- Mean number of skills per posting: **12.76**
- Median: **13**
- Interquartile range: **10 to 15**
- Minimum retained: **3**
- Maximum retained: **20**

The notebook also exports raw and cleaned skill-frequency tables to Excel, which is valuable for stakeholder review. Non-technical reviewers can inspect the vocabulary and challenge cleaning rules if necessary, making the process more governable.

### 3.6 Output Structure and Reusability

The cleaned dataset is saved as `data/cleaned_data/jobs_cleaned.pkl`. Before saving, the notebook drops intermediate helper columns and reorders the final schema so downstream consumers receive a compact, consistent table.

This is good execution practice. Instead of passing along every temporary artifact created during cleaning, the notebook separates internal processing columns from production-facing outputs. That makes later analysis cleaner and reduces accidental dependency on unstable intermediate fields.

### 3.7 Descriptive Validation and Exploratory Analysis

The second half of the notebook performs descriptive analysis on the cleaned data. This is not merely exploratory; it acts as a validation layer. If the top titles, skill distributions, and data-role patterns were obviously implausible, that would signal a problem in the cleaning pipeline.

Examples from the cleaned dataset include:

- Most common entry-level titles: `warehouse assistant` (31), `admin assistant` (24), `administrative assistant` (20), `sales executive` (19), `accounts assistant` (19)
- Most common skills overall: `Team Player` (2,857), `Customer Service` (2,114), `Interpersonal Skills` (2,059), `Communication Skills` (1,718), `Microsoft Office` (1,677)
- Titles with the widest skill range include `business development executive` (110 unique skills) and `marketing executive` (99 unique skills)

The notebook also isolates a subset of data-related roles using keyword matching. This subset contains **29 postings**, with top skills including `SQL` (14), `Data Analysis` (12), `Python` (12), `Business Analysis` (10), and `Business Requirements` (10). Median salary in this subset is **5,000**, with most postings marked as full-time.

These summaries directly support the broader project objective. They show what employers actually ask for and create a bridge to course-side skill extraction. For a university or public-sector workforce unit, this is the dataset that can later be matched against curriculum content to identify alignment gaps.

### 3.8 University Course Cleaning Methodology

- Add a matching methodology subsection for `data_cleaning_university_merged.ipynb`.
- Describe the course-side data sources:
  - NUSMods API
  - NTU scraper outputs and department mapping
  - SUTD scraper outputs
- Explain how module descriptions were cleaned and standardized.
- Document the cleaned course schema:
  - `code`
  - `title`
  - `department`
  - `description`
  - `university`
  - skill-related fields stored in the cleaned PKL
- Explain how module-side skills were produced in the notebook:
  - `skills_embedding`
  - `hard_skills`
  - `soft_skills`
- Add university-side data-quality issues:
  - missing descriptions from NTU and SUTD
  - uneven metadata richness across universities
  - department or faculty inconsistencies

### Part II: General Pipeline

### 3.9 Downstream Baseline Pipeline

- Add a subsection describing the official general pipeline in `src/create_test/`.
- Explain the role of:
  - `create_test_datasets.py`
  - `build_canonical_skill_framework.py`
  - `extract_job_ssoc3_from_original.py`
  - `canonical_skill_mapper.py`
  - `align_module_job_canonical.py`
- State that this pipeline now starts from:
  - `data/cleaned_data/combined_courses_cleaned.pkl`
  - `data/cleaned_data/jobs_cleaned.pkl`
- Explain the design decision to treat notebook-cleaned PKLs as the source of truth.
- Mention the shell shortcut:
  - `bash src/create_test/run_baseline_pipeline.sh`

### 3.10 Experimental Comparison Pipeline

- Add a subsection for the supported experimental path in `src/create_test/experimental/`.
- Explain what is being compared:
  - notebook-derived module skills versus independently extracted module skills
- Clarify what stays fixed during the comparison:
  - same module rows
  - same job-side canonical outputs
  - same canonical framework
- Explain why this comparison is useful:
  - isolates the effect of module skill extraction strategy
  - tests robustness of the alignment findings
- Mention the shell shortcut:
  - `bash src/create_test/run_experimental_pipeline.sh`

### Part III: STEM-Focused Pipeline

### 3.11 STEM Pipeline

- Add a subsection for the STEM-focused pipeline in `src/stem_test/`.
- Explain why the STEM branch exists:
  - narrower scope
  - stronger focus on technically oriented module-job alignment
  - ability to compare STEM-only alignment patterns against the broader baseline
- Describe the active STEM steps:
  - STEM scope classification
  - STEM-only dataset creation
  - independent module skill extraction
  - job SSOC enrichment
  - canonical skill mapping
  - alignment
- Note the design choice that the STEM pipeline is now PKL-first.
- Mention the shell shortcut:
  - `bash src/stem_test/run_stem_full_pipeline.sh`

### 3.12 Canonical Skill Framework

- Add a subsection explaining what the canonical framework is and why it is needed.
- Explain the problem it solves:
  - lexical variation across job and module skills
  - the need for a shared skill vocabulary before alignment
- Describe what is stored in the framework:
  - canonical skill label
  - skill type
  - aliases
  - excluded phrases
- State that the framework is now shared across the baseline and STEM pipelines.
- Explain why centralising it improves consistency and reproducibility.

### 3.13 Alignment Methodology

- Add a subsection explaining how module-job alignment is computed.
- Describe the use of canonical skill overlaps and job-group aggregation.
- Explain the role of SSOC grouping in structuring job demand.
- Summarise the scoring logic in plain language:
  - overlap
  - coverage
  - weighted similarity
  - gap interpretation
- Explain what the final output means for stakeholders:
  - indicative alignment, not causal proof
  - useful for curriculum review and prioritisation

### 3.14 Reproducibility and Repository Design

- Add a subsection documenting the repo cleanup and standardisation work.
- Explain how the repo is organized into:
  - baseline
  - experimental
  - STEM
  - legacy
- Mention the addition of shell-script shortcuts.
- Explain why this matters:
  - easier onboarding
  - easier reruns
  - clearer distinction between supported and exploratory code paths
- Mention any validation performed:
  - dry runs
  - pipeline output checks
  - consistency checks across frameworks

## 5. Findings and Evaluation 

### 4.1 Robustness

The notebook performs strongly on robustness.

- It handles nested, inconsistent JSON structures through explicit extraction functions rather than ad hoc one-off parsing.
- It accounts for multiple forms of data quality problems: HTML contamination, missing values, duplication, noisy skill labels, weak descriptions, and unknown employment metadata.
- It includes a meaningful imputation strategy for work type instead of silently discarding all partially incomplete records.
- It creates auditable artifacts, including Excel exports for raw and cleaned skill frequencies.

From a public-sector perspective, the strongest robustness feature is its domain-aware filtering. The notebook does not treat all job postings as equally relevant. It explicitly models the difference between fresh-graduate opportunities and other labour-market segments. That is essential when outputs may influence curriculum review or manpower policy discussions.

### 4.2 Execution

There are, however, still limitations:

These do not undermine the core cleaning pipeline, but they are important if the notebook is intended to serve as a polished production artifact.

- Add a subsection evaluating execution for the full project, not only the jobs notebook:
  - code organisation
  - reproducibility
  - readability
  - documentation
  - pipeline standardisation
- Mention the creation of shell shortcuts and clearer folder structure.
- Mention which parts of the project are now officially supported versus legacy.
- Explain how the final repository design improves maintainability and handoff.

### 4.3 Communication

- Add an explicit communication subsection aligned with the rubric.
- Evaluate:
  - whether the outputs are interpretable
  - whether the pipeline is understandable to a new user
  - whether the README and technical report are clear enough for both technical and non-technical readers
- Reference visual aids or propose visual aids that should appear in the report:
  - pipeline diagram
  - data attrition chart
  - alignment summary table
  - baseline vs experimental comparison table

### 4.4 Project Findings

- Add the actual end-to-end findings of the project here.
- Suggested points to include:
  - what the baseline alignment results suggest
  - what types of modules align well with job demand
  - where likely skill gaps appear
  - what the STEM-focused analysis shows
  - whether the experimental extractor materially changes the results
- Translate these findings into stakeholder-relevant takeaways:
  - curriculum review
  - employability programming
  - areas requiring deeper manual validation

### 4.5 Policy and Stakeholder Implications

- Add a subsection that connects findings to public-sector decision-making.
- Explain how ministries, universities, and workforce agencies could use the outputs.
- Clarify what decisions the project can support and what decisions it cannot support on its own.
- Note that the outputs are best treated as evidence for prioritisation and review, not automatic policy prescriptions.

## 6. Limitations, Biases, and Ethical Considerations

Several limitations should be stated explicitly.

First, the graduate filter is rule-based. Using `minimum_years_experience` in `{0,1}` is practical, but some graduate-suitable jobs may require 2 years, while some 0-1 year roles may still be unsuitable for typical undergraduates.

Second, postgraduate-role exclusion relies on keyword patterns in titles and descriptions. This improves precision, but it may still generate both false positives and false negatives.

Third, skill extraction depends on employer-supplied structured skill fields. Employers vary widely in how carefully they populate these fields. As a result, common soft skills may be overrepresented, while some technical competencies may be missing from the structured list even when present in the description text.

Fourth, the data-role subset is small at 29 postings. It is useful for illustration, but not yet strong enough for high-confidence sectoral conclusions.

Fifth, the notebook supports public-sector analysis but does not by itself resolve fairness concerns. For example, if certain industries systematically omit salary data or structured skills, the cleaned dataset may underrepresent them in downstream comparisons. Policymakers should treat the outputs as directional evidence rather than ground truth.

Additional limitations to document:

- The university-side dataset may not fully capture teaching quality, learning outcomes, or pedagogical depth; it mainly captures textual module descriptions and extracted skills.
- Canonical skill mapping introduces its own abstraction layer, which may merge distinct competencies or preserve distinctions that are not meaningful to employers.
- Alignment scores are similarity-based and should not be interpreted as causal measures of programme effectiveness.
- The STEM scope classification is rule-based and inherits the limitations of department-level labeling.
- Changes in labour-market language over time may reduce comparability if the framework is not periodically refreshed.

Ethical considerations to add:

- Explain the risk of overinterpreting employer language as objective labour-market truth.
- Note that course-job alignment should not be the only basis for judging educational value.
- Acknowledge the risk that humanities or interdisciplinary programmes may look weaker under a purely skill-overlap framing.
- Emphasise the importance of human review before using the outputs for high-stakes policy decisions.

## 7. Future areas for improvement

- Add concrete next steps for future project work:
  - improve fresh-graduate scoping heuristics beyond years-of-experience filtering alone
  - validate alignment outputs with expert review
  - expand the canonical framework iteratively using feedback
  - incorporate richer description-based skill extraction on the job side
  - add stronger evaluation metrics for baseline versus experimental skill extraction
  - extend analysis to trends over time or sector-specific substudies
  - incorporate more universities or broader education pathways if relevant
- Include future engineering improvements:
  - automated tests
  - versioned reference artifacts
  - stronger notebook-to-pipeline validation checks

## 8. Conclusion

- Add a short closing section that returns to the main project question.
- Summarise:
  - what data assets were built
  - what methodology was used
  - what the project contributes to university-job alignment analysis
- End with a balanced takeaway:
  - the project provides a robust, interpretable starting point for evidence-based curriculum review
  - but the outputs should be complemented by domain expertise and policy judgment

## 9. Suggested Figures and Tables

- Add a planning section for visuals if the final report will include them.
- Suggested visuals:
  - end-to-end pipeline diagram
  - job cleaning attrition table or waterfall chart
  - university data source summary table
  - canonical skill framework diagram
  - baseline versus experimental comparison table
  - STEM versus general pipeline comparison table
  - final alignment summary chart
