REVISION_SUFFIX = {
    3: "Fix only the flagged shots. All other shots must remain exactly as previously generated.",
    5: "Fix only the flagged VO lines. Do not alter any silent windows or lines not listed above.",
    6: "Fix only the flagged dialogue lines. Do not alter any lines not listed above.",
    7: "Fix only the flagged audio directives. Do not alter any windows not listed above.",
}


def build_revision_prompt_prefix(cycle: int,
                                  target_agent: int,
                                  instructions: list[dict]) -> str:
    lines = [
        f"REVISION MODE — Re-run #{cycle}.",
        "The following issues from your previous output MUST be corrected:\n"
    ]
    for inst in instructions:
        window      = inst.get('window_id') or 'N/A'
        dim         = inst.get('dimension') or inst.get('source') or 'general'
        instruction = inst.get('instruction', '')
        lines.append(f"- Window {window}: [{dim}] {instruction}")
    suffix = REVISION_SUFFIX.get(target_agent, "Fix only the flagged issues. Do not alter anything else.")
    lines.append(f"\n{suffix}")
    return "\n".join(lines)
