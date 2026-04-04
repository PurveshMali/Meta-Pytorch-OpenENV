"""
Email Triage Environment - Email Dataset
Realistic email scenarios with ground-truth labels for 3 tasks of increasing difficulty.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task 1 (EASY): Simple priority labelling
# 8 emails, clear signals, agent must label priority + category
# Max steps: 16 (2 actions per email typical)
# ---------------------------------------------------------------------------

TASK1_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "t1_e1",
        "sender": "ceo@company.com",
        "sender_domain": "company.com",
        "subject": "URGENT: Board presentation moved to tomorrow 9 AM",
        "body": (
            "Hi team,\n\nThe board meeting has been rescheduled to tomorrow 9 AM. "
            "Please ensure all slides are ready by 8 AM. This is time-sensitive.\n\nThanks"
        ),
        "timestamp": "2024-01-15T07:30:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "urgent",
        "true_category": "internal",
        "requires_reply": False,
    },
    {
        "email_id": "t1_e2",
        "sender": "noreply@newsletter.io",
        "sender_domain": "newsletter.io",
        "subject": "Your weekly digest is ready",
        "body": (
            "Click here to read this week's top stories in tech and business. "
            "Unsubscribe at any time."
        ),
        "timestamp": "2024-01-15T08:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "newsletter",
        "requires_reply": False,
    },
    {
        "email_id": "t1_e3",
        "sender": "john.smith@bigclient.com",
        "sender_domain": "bigclient.com",
        "subject": "Proposal for Q1 contract renewal - $500k",
        "body": (
            "Hello,\n\nWe are ready to discuss renewal of our contract worth $500,000 for Q1. "
            "Please have someone from your sales team reach out by Friday.\n\nBest,\nJohn"
        ),
        "timestamp": "2024-01-15T09:00:00",
        "has_attachment": True,
        "thread_length": 3,
        "true_priority": "high",
        "true_category": "sales_inquiry",
        "requires_reply": True,
    },
    {
        "email_id": "t1_e4",
        "sender": "billing@vendor.com",
        "sender_domain": "vendor.com",
        "subject": "Invoice #4521 - Payment Due",
        "body": (
            "Dear Customer,\n\nYour invoice #4521 for $1,200 is due on Jan 20. "
            "Please process payment at your earliest convenience.\n\nRegards,\nAccounting Team"
        ),
        "timestamp": "2024-01-15T10:00:00",
        "has_attachment": True,
        "thread_length": 1,
        "true_priority": "normal",
        "true_category": "billing",
        "requires_reply": False,
    },
    {
        "email_id": "t1_e5",
        "sender": "no-reply@spam-offers.xyz",
        "sender_domain": "spam-offers.xyz",
        "subject": "You have won $1,000,000! Claim now!!!",
        "body": (
            "Congratulations!!! You are our lucky winner. Send us your bank details "
            "to claim your prize. Limited time offer!!!"
        ),
        "timestamp": "2024-01-15T10:30:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "spam",
        "true_category": "spam",
        "requires_reply": False,
    },
    {
        "email_id": "t1_e6",
        "sender": "alice@team.company.com",
        "sender_domain": "company.com",
        "subject": "Team lunch this Friday?",
        "body": (
            "Hey everyone,\n\nShould we do team lunch on Friday? Pizza place on 5th Ave? "
            "Let me know your thoughts!\n\nAlice"
        ),
        "timestamp": "2024-01-15T11:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "internal",
        "requires_reply": True,
    },
    {
        "email_id": "t1_e7",
        "sender": "support@customer.net",
        "sender_domain": "customer.net",
        "subject": "Production system is DOWN - need help NOW",
        "body": (
            "Hi,\n\nOur production environment went down 20 minutes ago and we are losing "
            "revenue. Error: 503 service unavailable. We need immediate assistance.\n\nThanks"
        ),
        "timestamp": "2024-01-15T11:30:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "urgent",
        "true_category": "customer_support",
        "requires_reply": True,
    },
    {
        "email_id": "t1_e8",
        "sender": "dev@partner.org",
        "sender_domain": "partner.org",
        "subject": "API documentation question",
        "body": (
            "Hello,\n\nI was reviewing your API docs and have a question about rate limiting. "
            "What is the burst limit for the /v2/search endpoint?\n\nThanks,\nDev Team"
        ),
        "timestamp": "2024-01-15T12:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "normal",
        "true_category": "technical",
        "requires_reply": True,
    },
]

TASK1_CONFIG = {
    "task_id": "task1_easy_labelling",
    "description": "Label 8 emails with correct priority and category",
    "difficulty": "easy",
    "max_steps": 20,
    "emails": TASK1_EMAILS,
    "scoring": {
        "priority_weight": 0.6,
        "category_weight": 0.4,
    },
}

# ---------------------------------------------------------------------------
# Task 2 (MEDIUM): Triage + Reply drafting
# 6 emails, agent must both label AND write appropriate replies for emails
# that require them. Reply quality scored by keyword matching.
# Max steps: 18
# ---------------------------------------------------------------------------

TASK2_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "t2_e1",
        "sender": "angry.customer@hotmail.com",
        "sender_domain": "hotmail.com",
        "subject": "DISGUSTED with your service - cancelling subscription",
        "body": (
            "I have been a customer for 3 years and your latest update broke everything. "
            "My reports no longer export correctly. Fix this immediately or I am cancelling "
            "my $299/month subscription. I am furious."
        ),
        "timestamp": "2024-01-16T08:15:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "urgent",
        "true_category": "customer_support",
        "requires_reply": True,
        "reply_keywords": ["apologize", "sorry", "fix", "help", "resolve", "escalate", "priority"],
        "reply_avoid": ["unfortunately", "can't", "unable", "not our fault"],
    },
    {
        "email_id": "t2_e2",
        "sender": "procurement@enterprise.com",
        "sender_domain": "enterprise.com",
        "subject": "RFP for enterprise license - 500 seats",
        "body": (
            "Hello,\n\nWe are evaluating productivity tools for our 500-person engineering team. "
            "We need pricing, SLA details, and a demo scheduled before Jan 31. "
            "Budget is approved. Please respond ASAP.\n\nMaria Chen, VP Engineering"
        ),
        "timestamp": "2024-01-16T09:00:00",
        "has_attachment": True,
        "thread_length": 1,
        "true_priority": "high",
        "true_category": "sales_inquiry",
        "requires_reply": True,
        "reply_keywords": ["demo", "pricing", "enterprise", "schedule", "contact", "sales"],
        "reply_avoid": ["busy", "later", "wait"],
    },
    {
        "email_id": "t2_e3",
        "sender": "payroll@company.com",
        "sender_domain": "company.com",
        "subject": "Payroll discrepancy - please review",
        "body": (
            "Hi,\n\nWe noticed an overpayment of $3,400 in last month's payroll run. "
            "Please review the attached reconciliation report and confirm next steps "
            "for recovery by Thursday EOD.\n\nPayroll Team"
        ),
        "timestamp": "2024-01-16T10:00:00",
        "has_attachment": True,
        "thread_length": 2,
        "true_priority": "high",
        "true_category": "billing",
        "requires_reply": True,
        "reply_keywords": ["review", "confirm", "received", "Thursday", "will", "team"],
        "reply_avoid": [],
    },
    {
        "email_id": "t2_e4",
        "sender": "marketing@promo-blast.com",
        "sender_domain": "promo-blast.com",
        "subject": "Grow your business with our SEO package",
        "body": (
            "Hi there! Looking to get more traffic? Our SEO package starts at just $99/month. "
            "Click to see our plans. Unsubscribe if you don't want these emails."
        ),
        "timestamp": "2024-01-16T11:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "newsletter",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
    {
        "email_id": "t2_e5",
        "sender": "security@company.com",
        "sender_domain": "company.com",
        "subject": "ACTION REQUIRED: Suspicious login attempt on your account",
        "body": (
            "We detected a login attempt from an unrecognized device in Lagos, Nigeria "
            "at 3:47 AM. If this was not you, please reset your password immediately "
            "and contact the security team. Account ID: USR-88421."
        ),
        "timestamp": "2024-01-16T04:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "urgent",
        "true_category": "internal",
        "requires_reply": True,
        "reply_keywords": ["not me", "reset", "secure", "report", "investigate", "confirm", "yes"],
        "reply_avoid": [],
    },
    {
        "email_id": "t2_e6",
        "sender": "intern@company.com",
        "sender_domain": "company.com",
        "subject": "Question about onboarding documentation",
        "body": (
            "Hi,\n\nI am a new intern starting Monday and I am not sure where to find "
            "the onboarding checklist. Could you point me in the right direction?\n\nThanks,\nAlex"
        ),
        "timestamp": "2024-01-16T14:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "normal",
        "true_category": "internal",
        "requires_reply": True,
        "reply_keywords": ["onboarding", "checklist", "link", "wiki", "confluence", "welcome", "help"],
        "reply_avoid": [],
    },
]

TASK2_CONFIG = {
    "task_id": "task2_medium_triage_reply",
    "description": "Triage 6 emails AND draft quality replies for emails that need them",
    "difficulty": "medium",
    "max_steps": 18,
    "emails": TASK2_EMAILS,
    "scoring": {
        "priority_weight": 0.35,
        "category_weight": 0.25,
        "reply_quality_weight": 0.40,
    },
}

# ---------------------------------------------------------------------------
# Task 3 (HARD): Full inbox management under pressure
# 10 emails with ambiguous signals, contradictory context, time sensitivity.
# Agent must label + route + reply with nuance. Many emails have similar
# surface features but different true priorities.
# Max steps: 30
# ---------------------------------------------------------------------------

TASK3_EMAILS: List[Dict[str, Any]] = [
    {
        "email_id": "t3_e1",
        "sender": "cfo@company.com",
        "sender_domain": "company.com",
        "subject": "Quick question",
        "body": (
            "Hey, when you get a chance, can you send me last quarter's customer churn numbers? "
            "No rush, this week is fine."
        ),
        "timestamp": "2024-01-17T08:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "high",   # CFO request, even if casual tone
        "true_category": "internal",
        "requires_reply": True,
        "reply_keywords": ["churn", "numbers", "report", "send", "will", "today", "week"],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e2",
        "sender": "alert@monitoring-service.com",
        "sender_domain": "monitoring-service.com",
        "subject": "CRITICAL: CPU usage at 98% - auto-scaled",
        "body": (
            "Automated alert: Server cluster US-EAST-1 CPU usage reached 98% at 07:53 UTC. "
            "Auto-scaling triggered. 3 new instances spun up. No action required unless "
            "this alert continues."
        ),
        "timestamp": "2024-01-17T07:55:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "normal",  # Auto-handled, flagged as critical but resolved
        "true_category": "technical",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e3",
        "sender": "lawyer@legalfirm.com",
        "sender_domain": "legalfirm.com",
        "subject": "Re: Data processing agreement - final comments",
        "body": (
            "Please find our final comments on the DPA attached. We need your legal team's "
            "sign-off by Jan 19 or the partnership launch must be delayed. "
            "This is a hard deadline per our client's compliance team."
        ),
        "timestamp": "2024-01-17T09:00:00",
        "has_attachment": True,
        "thread_length": 7,
        "true_priority": "urgent",
        "true_category": "other",   # Legal/compliance doesn't fit neatly
        "requires_reply": True,
        "reply_keywords": ["legal", "review", "team", "deadline", "confirm", "received", "escalate"],
        "reply_avoid": ["delay", "wait", "later"],
    },
    {
        "email_id": "t3_e4",
        "sender": "noreply@github.com",
        "sender_domain": "github.com",
        "subject": "Your PR #1247 has been merged",
        "body": (
            "Pull request #1247 'Fix memory leak in cache layer' has been merged into main "
            "by @alice. 3 CI checks passed."
        ),
        "timestamp": "2024-01-17T09:30:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "technical",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e5",
        "sender": "john.doe@competitor-corp.com",
        "sender_domain": "competitor-corp.com",
        "subject": "Interesting opportunity to discuss",
        "body": (
            "Hi,\n\nI came across your profile and think there could be some interesting "
            "synergies between our organizations. Would you be open to a 30-min call "
            "to explore?\n\nJohn Doe\nVP Business Development, CompetitorCorp"
        ),
        "timestamp": "2024-01-17T10:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",   # Unsolicited BD outreach from competitor
        "true_category": "other",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e6",
        "sender": "bob@longtime-customer.com",
        "sender_domain": "longtime-customer.com",
        "subject": "Happy with the new feature!",
        "body": (
            "Just wanted to say the new dashboard export feature is fantastic. "
            "Saved us hours this week. Keep up the great work! No action needed from you."
        ),
        "timestamp": "2024-01-17T10:15:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "customer_support",
        "requires_reply": True,   # Good relationship management
        "reply_keywords": ["thank", "glad", "happy", "appreciate", "feedback"],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e7",
        "sender": "compliance@regulator.gov",
        "sender_domain": "regulator.gov",
        "subject": "Notice of Audit - Response Required Within 5 Business Days",
        "body": (
            "Dear Company Representative,\n\nThis notice informs you that your organization "
            "has been selected for a routine data handling compliance audit. You are required "
            "to submit documentation (listed in attachment) within 5 business days. "
            "Failure to comply may result in penalties.\n\nRegulator's Office"
        ),
        "timestamp": "2024-01-17T11:00:00",
        "has_attachment": True,
        "thread_length": 1,
        "true_priority": "urgent",
        "true_category": "other",
        "requires_reply": True,
        "reply_keywords": ["acknowledge", "received", "legal", "team", "comply", "document", "audit"],
        "reply_avoid": ["ignore", "later", "not sure"],
    },
    {
        "email_id": "t3_e8",
        "sender": "hr@company.com",
        "sender_domain": "company.com",
        "subject": "Performance review cycle starts Monday",
        "body": (
            "Hi all,\n\nA reminder that the Q4 performance review cycle begins Monday. "
            "Please complete your self-assessments in the HR portal by Jan 26. "
            "Manager reviews are due Feb 2.\n\nHR Team"
        ),
        "timestamp": "2024-01-17T12:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "normal",
        "true_category": "internal",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
    {
        "email_id": "t3_e9",
        "sender": "customer@startup.io",
        "sender_domain": "startup.io",
        "subject": "Feature request: bulk export",
        "body": (
            "Hey team,\n\nLove the product but we really need a bulk export feature. "
            "We're on the Pro plan and this is blocking us from fully migrating our workflow. "
            "Any ETA? We've been asking for 6 months and we may have to look at alternatives."
        ),
        "timestamp": "2024-01-17T13:00:00",
        "has_attachment": False,
        "thread_length": 4,
        "true_priority": "high",   # Churn risk on paying customer
        "true_category": "customer_support",
        "requires_reply": True,
        "reply_keywords": ["roadmap", "priority", "feedback", "team", "understand", "soon", "update"],
        "reply_avoid": ["no ETA", "can't promise", "not possible"],
    },
    {
        "email_id": "t3_e10",
        "sender": "ops@company.com",
        "sender_domain": "company.com",
        "subject": "Office WiFi maintenance window - Jan 18 2-4 AM",
        "body": (
            "Just a heads up: IT will perform WiFi maintenance on Jan 18 from 2-4 AM. "
            "Remote access via VPN will be unaffected. No action required."
        ),
        "timestamp": "2024-01-17T14:00:00",
        "has_attachment": False,
        "thread_length": 1,
        "true_priority": "low",
        "true_category": "internal",
        "requires_reply": False,
        "reply_keywords": [],
        "reply_avoid": [],
    },
]

TASK3_CONFIG = {
    "task_id": "task3_hard_full_inbox",
    "description": (
        "Full inbox management: 10 emails with ambiguous signals. "
        "Must correctly handle casual-toned urgent emails, auto-resolved alerts, "
        "regulatory notices, and churn-risk customers."
    ),
    "difficulty": "hard",
    "max_steps": 30,
    "emails": TASK3_EMAILS,
    "scoring": {
        "priority_weight": 0.35,
        "category_weight": 0.20,
        "reply_quality_weight": 0.30,
        "escalation_accuracy_weight": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS = {
    "task1_easy_labelling": TASK1_CONFIG,
    "task2_medium_triage_reply": TASK2_CONFIG,
    "task3_hard_full_inbox": TASK3_CONFIG,
}
