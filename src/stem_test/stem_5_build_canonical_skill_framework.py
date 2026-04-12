import argparse
import json
import re
from pathlib import Path
from typing import Optional

from module_skill_rules import (
    CANONICAL_MODULE_SKILLS,
    EXPLICIT_PHRASE_BLOCKLIST,
    MODULE_SKILL_RULES,
    STRICT_CANONICAL_EVIDENCE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reference" / "canonical_skill_framework_v4.json"

DEFAULT_EXCLUDED_PHRASES = {
    "attention to detail",
    "driving license",
    "class 3 driving license",
    "physically fit",
    "pressure",
    "wellbeing",
    "ability to multitask",
    "able to multitask",
    "approachable",
    "arranging",
}

BASE_SKILLS = [
    {
        "canonical_skill": "Communication",
        "skill_type": "soft",
        "aliases": ["communication", "communication skills", "verbal communication"],
        "notes": "General communication capability.",
    },
    {
        "canonical_skill": "Interpersonal Skills",
        "skill_type": "soft",
        "aliases": ["interpersonal skills"],
        "notes": "Relational and people-facing capability.",
    },
    {
        "canonical_skill": "Teamwork",
        "skill_type": "soft",
        "aliases": ["team player", "team collaboration", "teamwork", "ability to work independently"],
        "notes": "Collaboration-oriented behaviors.",
    },
    {
        "canonical_skill": "Leadership",
        "skill_type": "soft",
        "aliases": ["leadership", "team leader", "management skills"],
        "notes": "People leadership and coordination.",
    },
    {
        "canonical_skill": "Project Management",
        "skill_type": "soft",
        "aliases": ["project management", "project planning"],
        "notes": "Planning, execution, and delivery of projects.",
    },
    {
        "canonical_skill": "Time Management",
        "skill_type": "soft",
        "aliases": ["time management", "scheduling"],
        "notes": "Time and workload management.",
    },
    {
        "canonical_skill": "Customer Service",
        "skill_type": "soft",
        "aliases": ["customer service", "customer service skills", "customer satisfaction", "customer experience"],
        "notes": "General service quality and client interaction.",
    },
    {
        "canonical_skill": "Microsoft Office",
        "skill_type": "tool",
        "aliases": ["microsoft office", "ms office", "microsoft word", "microsoft powerpoint", "microsoft excel", "excel"],
        "notes": "Office productivity tools.",
    },
    {
        "canonical_skill": "Java",
        "skill_type": "tool",
        "aliases": ["java"],
        "notes": "Programming language.",
    },
    {
        "canonical_skill": "Statistics",
        "skill_type": "hard",
        "aliases": ["statistics", "statistical analysis", "probability", "econometrics", "regression analysis"],
        "notes": "Broad statistical methods bucket.",
    },
    {
        "canonical_skill": "Accounting",
        "skill_type": "hard",
        "aliases": ["accounting", "accounts payable", "accounts receivable", "financial reporting"],
        "notes": "Accounting and reporting tasks.",
    },
    {
        "canonical_skill": "Finance",
        "skill_type": "domain",
        "aliases": ["banking", "financial services", "financial analysis", "forecasting", "payroll", "cost control"],
        "notes": "Finance and financial operations.",
    },
    {
        "canonical_skill": "Risk Management",
        "skill_type": "hard",
        "aliases": ["risk management", "risk assessment", "compliance", "regulatory requirements"],
        "notes": "Governance, risk, and compliance.",
    },
    {
        "canonical_skill": "Marketing",
        "skill_type": "hard",
        "aliases": ["marketing", "digital marketing", "marketing strategy", "brand awareness", "advertising", "social media", "lead generation"],
        "notes": "Marketing and demand generation.",
    },
    {
        "canonical_skill": "Sales",
        "skill_type": "hard",
        "aliases": ["sales", "selling", "sales process", "account management", "business development", "pricing"],
        "notes": "Commercial and revenue-generating work.",
    },
    {
        "canonical_skill": "Human Resources",
        "skill_type": "domain",
        "aliases": ["human resources", "recruiting"],
        "notes": "People operations and talent processes.",
    },
    {
        "canonical_skill": "Supply Chain Management",
        "skill_type": "hard",
        "aliases": ["supply chain", "supply chain management", "procurement", "purchasing", "sourcing", "shipping", "warehousing", "inventory management", "transportation", "logistics planning"],
        "notes": "Supply chain, logistics, and procurement.",
    },
    {
        "canonical_skill": "Quality Assurance",
        "skill_type": "hard",
        "aliases": ["quality assurance", "quality control", "quality improvement", "iso", "audit"],
        "notes": "Quality and compliance processes.",
    },
]

TOOL_SKILLS = {
    "Microsoft Office",
    "Python",
    "SQL",
    "Java",
    "CAD Prototyping",
}

SOFT_SKILLS = {
    "Communication",
    "Interpersonal Skills",
    "Teamwork",
    "Leadership",
    "Problem Solving",
    "Critical Thinking",
    "Presentation Skills",
    "Research",
    "Project Management",
    "Time Management",
    "Coaching and Mentoring",
    "Customer Service",
    "Negotiation",
    "Stakeholder Management",
    "Administration",
    "Collaboration",
    "Learning Strategies",
}

DOMAIN_SKILLS = {
    "Finance",
    "Healthcare",
    "Life Science Research",
    "Human Resources",
    "Media Law",
    "Tax Law",
    "International Taxation",
    "Data Privacy Law",
    "Intellectual Property Law",
    "Charity Law",
    "Arbitration",
}


def normalize_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"^[^a-z0-9+#/]+|[^a-z0-9+#/]+$", "", cleaned)
    return cleaned


def infer_skill_type(skill: str, seed_type: Optional[str] = None) -> str:
    if seed_type:
        return seed_type
    if skill in TOOL_SKILLS:
        return "tool"
    if skill in SOFT_SKILLS:
        return "soft"
    if skill in DOMAIN_SKILLS:
        return "domain"
    return "hard"


def default_note(skill: str) -> str:
    return f"Canonical skill generated from module skill rules ({skill})."


def build_framework(version: str = "v4") -> dict:
    alias_map: dict[str, set[str]] = {}
    skill_type_map: dict[str, str] = {}
    note_map: dict[str, str] = {}

    for row in BASE_SKILLS:
        skill = row["canonical_skill"]
        aliases = {normalize_phrase(skill)}
        aliases.update(normalize_phrase(a) for a in row.get("aliases", []))
        aliases = {a for a in aliases if a}

        alias_map.setdefault(skill, set()).update(aliases)
        skill_type_map[skill] = row.get("skill_type", "hard")
        note_map[skill] = row.get("notes") or default_note(skill)

    for rule in MODULE_SKILL_RULES:
        patterns = [normalize_phrase(p) for p in rule.get("patterns", []) if normalize_phrase(p)]
        for skill in rule.get("skills", []):
            alias_map.setdefault(skill, set()).add(normalize_phrase(skill))
            alias_map[skill].update(patterns)
            skill_type_map.setdefault(skill, infer_skill_type(skill))
            note_map.setdefault(skill, default_note(skill))

    for skill, evidence in STRICT_CANONICAL_EVIDENCE.items():
        alias_map.setdefault(skill, set()).add(normalize_phrase(skill))
        alias_map[skill].update(normalize_phrase(e) for e in evidence if normalize_phrase(e))
        skill_type_map.setdefault(skill, infer_skill_type(skill))
        note_map.setdefault(skill, default_note(skill))

    for skill in CANONICAL_MODULE_SKILLS:
        alias_map.setdefault(skill, set()).add(normalize_phrase(skill))
        skill_type_map.setdefault(skill, infer_skill_type(skill))
        note_map.setdefault(skill, default_note(skill))

    skills = []
    for canonical_skill in sorted(alias_map):
        aliases = sorted(a for a in alias_map[canonical_skill] if a)
        skills.append(
            {
                "canonical_skill": canonical_skill,
                "skill_type": infer_skill_type(canonical_skill, skill_type_map.get(canonical_skill)),
                "aliases": aliases,
                "notes": note_map.get(canonical_skill, default_note(canonical_skill)),
            }
        )

    excluded = sorted(
        normalize_phrase(x)
        for x in (set(EXPLICIT_PHRASE_BLOCKLIST) | DEFAULT_EXCLUDED_PHRASES)
        if normalize_phrase(x)
    )

    framework = {
        "version": version,
        "description": (
            "Canonical skill framework for module-job alignment, generated from "
            "stem_test rule definitions and curated baseline occupational skills."
        ),
        "fields": {
            "canonical_skill": "Stable project-wide skill label.",
            "skill_type": "One of hard, soft, domain, or tool.",
            "aliases": "Observed phrases or close synonyms that should normalize into the canonical label.",
            "notes": "Optional explanation for scope or intended usage.",
        },
        "excluded_phrases": excluded,
        "skills": skills,
    }
    return framework


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical_skill_framework_v4.json from stem_test skill-rule code."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--version",
        default="v4",
        help="Framework version string to write in metadata (default: v4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build in-memory and print summary without writing file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    framework = build_framework(version=args.version)

    if args.dry_run:
        print(
            f"Built framework {framework['version']} with "
            f"{len(framework['skills'])} skills and "
            f"{len(framework['excluded_phrases'])} excluded phrases."
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(framework, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {args.output} with {len(framework['skills'])} skills "
        f"and {len(framework['excluded_phrases'])} excluded phrases."
    )


if __name__ == "__main__":
    main()
