"""Synthetic sample job-listing generator.

RoleMatch AI ships with generated demo listings so the search and agent are
fully exercisable without scraping real boards. Everything here is fictional
sample data - the README and the UI both say so. The generator is seeded, so
the catalog is reproducible.
"""

from __future__ import annotations

import random

COMPANIES = [
    "Zylo Labs", "Brightstack", "NimbusPay", "Quantico Systems", "Verdant AI",
    "PixelForge", "DataLoom", "CloudKart", "MediTrack", "FinArc",
    "Orbitware", "Synthio", "Kestrel Networks", "Bloomreach Labs", "Agrinova",
    "HexaCloud", "Trailblitz", "OmniCart", "Ledgerly", "Voyant Health",
    "Stratus Nine", "Cognitrix", "FleetIQ", "EduMentor", "RetailZen",
    "SignalPath", "UrbanGrid", "NovaBite", "Craftlane", "SecureNest",
    "AsterPay", "MellowDB", "QuillWorks", "TandemHR", "InsightLoop",
    "VoltEdge", "Papertrail AI", "GreenMile Tech", "BlueOrbit", "LumenCast",
]

LOCATIONS = [
    ("Bengaluru", 0.30), ("Hyderabad", 0.15), ("Pune", 0.12),
    ("Gurugram", 0.10), ("Mumbai", 0.08), ("Chennai", 0.08),
    ("Noida", 0.07), ("Remote - India", 0.10),
]

TRACKS = {
    "backend": {
        "titles": [
            "Backend Engineer", "Software Engineer - Backend", "API Platform Engineer",
            "Python Developer", "Node.js Backend Developer", "Distributed Systems Engineer",
        ],
        "skills": ["Python", "FastAPI", "Node.js", "PostgreSQL", "Redis", "Docker",
                   "REST APIs", "SQL", "Django", "Kafka", "AWS", "gRPC"],
    },
    "frontend": {
        "titles": [
            "Frontend Engineer", "React Developer", "UI Engineer",
            "Software Engineer - Web", "Next.js Developer",
        ],
        "skills": ["React", "TypeScript", "Next.js", "Tailwind CSS", "JavaScript",
                   "Redux", "HTML", "CSS", "Vite", "Accessibility"],
    },
    "fullstack": {
        "titles": [
            "Full Stack Engineer", "Software Development Engineer", "Product Engineer",
            "Full Stack Developer (MERN)",
        ],
        "skills": ["React", "Node.js", "TypeScript", "PostgreSQL", "MongoDB",
                   "Express", "Docker", "AWS", "GraphQL", "CI/CD"],
    },
    "ai": {
        "titles": [
            "AI Engineer", "LLM Engineer", "Applied AI Engineer", "ML Engineer",
            "GenAI Developer", "AI Agents Engineer",
        ],
        "skills": ["Python", "LangChain", "RAG", "Vector Databases", "OpenAI",
                   "Embeddings", "FastAPI", "Prompt Engineering", "PyTorch", "pgvector"],
    },
    "data": {
        "titles": [
            "Data Engineer", "Analytics Engineer", "Data Platform Engineer",
            "ETL Developer",
        ],
        "skills": ["Python", "SQL", "Airflow", "Spark", "dbt", "Snowflake",
                   "PostgreSQL", "Kafka", "AWS", "Pandas"],
    },
    "devops": {
        "titles": [
            "DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer",
            "Platform Engineer",
        ],
        "skills": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux",
                   "GitHub Actions", "Monitoring", "Python", "Bash"],
    },
    "qa": {
        "titles": [
            "QA Engineer", "SDET", "Test Automation Engineer",
        ],
        "skills": ["Selenium", "Python", "Pytest", "API Testing", "CI/CD",
                   "JavaScript", "Playwright", "SQL"],
    },
    "mobile": {
        "titles": ["Android Developer", "Flutter Developer", "React Native Engineer"],
        "skills": ["Kotlin", "Flutter", "Dart", "React Native", "Firebase",
                   "REST APIs", "TypeScript"],
    },
}

SENIORITY = [
    # (label, exp_min, exp_max, salary band LPA, weight)
    ("Intern", 0, 0, (1.5, 4), 0.10),
    ("Junior", 0, 1, (4, 9), 0.22),
    ("", 1, 3, (8, 16), 0.30),
    ("Senior", 3, 6, (16, 30), 0.24),
    ("Staff", 6, 10, (30, 55), 0.10),
    ("Lead", 5, 9, (25, 45), 0.04),
]

WORK_MODES = [("on-site", 0.42), ("hybrid", 0.33), ("remote", 0.25)]

DESCRIPTION_TEMPLATES = [
    "You will own {focus} for our {product} product, working closely with "
    "product and design to ship iteratively. Day to day you write {skill_a} and "
    "{skill_b}, review code, and raise the bar on reliability. We value "
    "engineers who communicate clearly and care about user outcomes.",
    "We are hiring for the {product} team. You will build and operate {focus}, "
    "using {skill_a}, {skill_b} and {skill_c}. The team ships weekly, writes "
    "design docs for non-trivial changes, and runs blameless postmortems.",
    "As a {title} you will design, build and measure {focus}. Expect a mix of "
    "greenfield work and careful refactoring across {skill_a} and {skill_b}. "
    "You will mentor peers through reviews and help shape our roadmap.",
    "Our {product} squad needs someone to drive {focus}. You should be "
    "comfortable with {skill_a}, {skill_b} and production systems. We offer "
    "high ownership, fast feedback loops, and a pragmatic engineering culture.",
]

PRODUCTS = ["payments", "search", "messaging", "analytics", "growth",
            "core platform", "developer tools", "marketplace", "mobile",
            "recommendations"]

FOCI = ["backend services and APIs", "user-facing web experiences",
        "data pipelines and storage", "internal tooling",
        "AI-powered product features", "reliability and infrastructure",
        "end-to-end product features", "integrations with external systems"]


def _weighted(rng: random.Random, items):
    values = [i[0] for i in items]
    weights = [i[1] for i in items]
    return rng.choices(values, weights=weights, k=1)[0]


def generate_jobs(count: int, seed: int = 42, offset: int = 0) -> list[dict]:
    """Generate `count` deterministic synthetic listings."""
    rng = random.Random(seed + offset)
    jobs = []
    for _ in range(count):
        track_name = _weighted(rng, [(k, 1.0) for k in TRACKS])
        track = TRACKS[track_name]
        title_base = rng.choice(track["titles"])
        label, exp_min, exp_max, (sal_lo, sal_hi), _ = SENIORITY[
            rng.choices(range(len(SENIORITY)), weights=[s[4] for s in SENIORITY], k=1)[0]
        ]
        title = f"{label} {title_base}".strip()
        company = rng.choice(COMPANIES)
        location = _weighted(rng, LOCATIONS)
        work_mode = "remote" if location.startswith("Remote") else _weighted(rng, WORK_MODES)

        n_skills = rng.randint(4, 7)
        skills = rng.sample(track["skills"], min(n_skills, len(track["skills"])))

        salary_min = round(rng.uniform(sal_lo, sal_lo + (sal_hi - sal_lo) * 0.4), 1)
        salary_max = round(salary_min + rng.uniform((sal_hi - sal_lo) * 0.3, (sal_hi - sal_lo) * 0.8), 1)
        if label == "Intern":
            salary_min, salary_max = 0.5, round(rng.uniform(1.5, 4.0), 1)

        description = rng.choice(DESCRIPTION_TEMPLATES).format(
            focus=rng.choice(FOCI),
            product=rng.choice(PRODUCTS),
            title=title_base,
            skill_a=skills[0],
            skill_b=skills[1] if len(skills) > 1 else skills[0],
            skill_c=skills[2] if len(skills) > 2 else skills[0],
        )
        description += (
            f" This is a {work_mode} role based in {location}. "
            f"Compensation band: {salary_min}-{salary_max} LPA."
        )

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "work_mode": work_mode,
                "experience_min": float(exp_min),
                "experience_max": float(exp_max),
                "salary_min_lpa": salary_min,
                "salary_max_lpa": salary_max,
                "skills": skills,
                "description": description,
                "posted_days_ago": rng.randint(0, 30),
            }
        )
    return jobs


def embed_text_for(job: dict) -> str:
    """The document we embed for semantic search."""
    return (
        f"{job['title']} at {job['company']} in {job['location']} ({job['work_mode']}). "
        f"Skills: {', '.join(job['skills'])}. "
        f"Experience {job['experience_min']}-{job['experience_max']} years. "
        f"{job['description']}"
    )
