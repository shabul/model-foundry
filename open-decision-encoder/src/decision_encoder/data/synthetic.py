"""Deterministic policies, varied renderings and auditable labels; no LLM required."""

import hashlib
import json
import random

from decision_encoder.data.schema import ABSTAIN_OPTION, split_group

# Policy text is part of the input: arbitrary thresholds are never hidden rules.
DOMAINS = {
    "tool_routing": (
        "Which tool should handle this request?",
        ["Search", "Database", "Calculator", "Ask for clarification"],
        "Use Search for external facts, Database for internal records, Calculator for arithmetic, and Ask for clarification when the request is unclear.",
    ),
    "agent_action_selection": (
        "What should the agent do next?",
        ["Execute", "Request approval", "Gather evidence", "Stop"],
        "Stop if cancelled; otherwise Gather evidence if evidence is missing; otherwise Request approval if approval is required and absent; otherwise Execute.",
    ),
    "incident_triage": (
        "How should this incident be handled?",
        ["Monitor", "Investigate", "Contain", "Close"],
        "Contain confirmed active compromise; otherwise Investigate suspicious activity; otherwise Close a resolved incident; otherwise Monitor.",
    ),
    "invoice_processing": (
        "What should happen to this invoice?",
        ["Pay", "Hold", "Reject", "Request documents"],
        "Reject duplicate invoices; otherwise Request documents if proof of delivery is absent; otherwise Hold if amounts do not match; otherwise Pay.",
    ),
    "workflow_routing": (
        "Where should this work item go?",
        ["Engineering", "Billing", "Legal", "Support"],
        "Send defects to Engineering, payment disputes to Billing, contract reviews to Legal, and general requests to Support.",
    ),
    "model_routing": (
        "Which model route should be used?",
        ["Local", "Fast", "Reasoning", "Fallback"],
        "Use Local for private data; otherwise Fallback if the primary provider is down; otherwise Reasoning for complex problems; otherwise Fast.",
    ),
    "support_ticket_routing": (
        "Which team should receive this ticket?",
        ["Access", "Payments", "Delivery", "Technical"],
        "Route login issues to Access, charges to Payments, shipments to Delivery, and software errors to Technical.",
    ),
    "risk_severity": (
        "What is the risk severity?",
        ["Low", "Medium", "High", "Critical"],
        "Severity is Critical if exposed accounts are at least 1000, High if at least 100, Medium if at least 10, and Low otherwise.",
    ),
    "retry_fallback": (
        "What should the gateway do next?",
        ["Return result", "Retry", "Use fallback", "Abort"],
        "Return result on success; otherwise Retry if attempts are below 2; otherwise Use fallback if available; otherwise Abort.",
    ),
    "verification": (
        "Did the operation meet all verification requirements?",
        ["Yes", "No"],
        "Yes requires both a valid signature and a matching checksum; otherwise No.",
    ),
}


def policy_state(domain, rng):
    def b():
        return bool(rng.randrange(2))

    if domain == "tool_routing":
        kind = rng.choice(["external facts", "internal records", "arithmetic", "unclear"])
        return {"request_kind": kind}, ["external facts", "internal records", "arithmetic", "unclear"].index(
            kind
        )
    if domain == "agent_action_selection":
        s = dict(cancelled=b(), evidence_present=b(), approval_required=b(), approval_present=b())
        return s, 3 if s["cancelled"] else 2 if not s["evidence_present"] else 1 if s[
            "approval_required"
        ] and not s["approval_present"] else 0
    if domain == "incident_triage":
        s = dict(active_compromise=b(), suspicious_activity=b(), resolved=b())
        return s, 2 if s["active_compromise"] else 1 if s["suspicious_activity"] else 3 if s[
            "resolved"
        ] else 0
    if domain == "invoice_processing":
        s = dict(duplicate=b(), delivery_proof=b(), amounts_match=b())
        return s, 2 if s["duplicate"] else 3 if not s["delivery_proof"] else 1 if not s[
            "amounts_match"
        ] else 0
    if domain in {"workflow_routing", "support_ticket_routing"}:
        values = (
            ["defect", "payment dispute", "contract review", "general request"]
            if domain == "workflow_routing"
            else ["login", "charge", "shipment", "software error"]
        )
        value = rng.choice(values)
        return {"request_type": value}, values.index(value)
    if domain == "model_routing":
        s = dict(private_data=b(), primary_down=b(), complex_problem=b())
        return s, 0 if s["private_data"] else 3 if s["primary_down"] else 2 if s["complex_problem"] else 1
    if domain == "risk_severity":
        n = rng.choice(
            [rng.randrange(10), rng.randrange(10, 100), rng.randrange(100, 1000), rng.randrange(1000, 5000)]
        )
        return {"exposed_accounts": n}, 3 if n >= 1000 else 2 if n >= 100 else 1 if n >= 10 else 0
    if domain == "retry_fallback":
        s = dict(
            status=rng.choice(["success", "timeout", "error"]),
            attempts=rng.randrange(5),
            fallback_available=b(),
        )
        return s, 0 if s["status"] == "success" else 1 if s["attempts"] < 2 else 2 if s[
            "fallback_available"
        ] else 3
    s = dict(signature_valid=b(), checksum_matches=b())
    return s, 0 if s["signature_valid"] and s["checksum_matches"] else 1


def generate_synthetic(count=30000, seed=42, reserve_test=False):
    rng = random.Random(seed)
    domains = list(DOMAINS)
    for i in range(count):
        domain = domains[i % len(domains)]
        question, labels, policy = DOMAINS[domain]
        structured, target = policy_state(domain, rng)
        # Group on causal state, excluding irrelevant identifiers and surface form.
        key = domain + json.dumps(structured, sort_keys=True)
        group = hashlib.sha256(key.encode()).hexdigest()
        options = [
            {
                "id": label.lower().replace(" ", "_"),
                "label": label,
                "description": f"Select {label} according to the stated policy.",
            }
            for label in labels
        ]
        kind = "ordinal" if domain == "risk_severity" else "boolean" if domain == "verification" else "choice"
        if kind == "ordinal":
            for j, option in enumerate(options):
                option["value"] = j
        reason = None
        # ~12% positives, plus ~35% negative abstention candidates, restricted to choices.
        roll = rng.random()
        if kind == "choice" and roll < 0.16:
            if rng.random() < 0.5:
                options.pop(target)
                reason = "gold_removed"
            else:
                structured = {k: "unknown" for k in structured}
                reason = "missing_evidence"
            options.append(dict(ABSTAIN_OPTION))
            target = len(options) - 1
        elif kind == "choice" and roll < 0.60:
            options.append(dict(ABSTAIN_OPTION))
        pairs = list(structured.items())
        rng.shuffle(pairs)
        template = rng.randrange(3)
        if template == 0:
            body = "\n".join(f"{k} = {str(v).lower()}" for k, v in pairs)
        elif template == 1:
            body = " ".join(f"The {k.replace('_', ' ')} is {str(v).lower()}." for k, v in pairs)
        else:
            body = json.dumps(dict(pairs), sort_keys=False)
        state = f"Policy: {policy}\nObserved facts: {body}\nUnrelated tracking reference: R{rng.randrange(1000000)}."
        y = [float(j == target) for j in range(len(options))]
        order = list(range(len(options)))
        rng.shuffle(order)
        # Unknown states group together, not with their hidden original labels.
        if reason == "missing_evidence":
            group = hashlib.sha256((domain + ":missing").encode()).hexdigest()
        split = split_group(group, seed)
        if reserve_test and int(hashlib.sha256(("test:" + group).encode()).hexdigest()[:8], 16) % 10 == 0:
            split = "test"
        yield {
            "id": f"synthetic-{seed}-{i}",
            "source": "synthetic",
            "source_split": "test" if split == "test" else "train",
            "group_id": group,
            "decision_type": kind,
            "state": state,
            "question": question,
            "options": [options[j] for j in order],
            "target_probabilities": [y[j] for j in order],
            "metadata": {
                "seed": seed,
                "domain": domain,
                "rule": policy,
                "generator_version": 1,
                "structured_state": structured,
                "template": template,
                "difficulty": "policy",
                "abstention_reason": reason,
                "split": split,
            },
        }
