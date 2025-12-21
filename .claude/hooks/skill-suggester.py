#!/usr/bin/env python3
"""
Smart skill suggester for Claude Code.
Analyzes user prompts and file context to suggest relevant skills.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional


class SkillSuggester:
    def __init__(self, rules_path: str):
        """Initialize the skill suggester with rules from JSON."""
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()
        self.session_skills_used = set()

    def _load_rules(self) -> Dict:
        """Load skill rules from JSON file."""
        try:
            with open(self.rules_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading skill rules: {e}", file=sys.stderr)
            return {"skills": {}}

    def _check_skip_conditions(self, skill_name: str, skill_config: Dict, file_paths: List[str]) -> bool:
        """Check if skill should be skipped based on skip conditions."""
        skip_conditions = skill_config.get("skipConditions", {})

        # Check if skill was already used in this session
        if skip_conditions.get("sessionSkillUsed", False):
            if skill_name in self.session_skills_used:
                return True

        # Check environment override
        env_override = skip_conditions.get("envOverride")
        if env_override and os.getenv(env_override):
            return True

        # Check file markers (skip validation comments in files)
        file_markers = skip_conditions.get("fileMarkers", [])
        if file_markers and file_paths:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read(1000)  # Check first 1000 chars
                            for marker in file_markers:
                                if marker in content:
                                    return True
                    except:
                        pass

        return False

    def _match_keywords(self, prompt: str, keywords: List[str]) -> int:
        """Count keyword matches in prompt (case-insensitive)."""
        prompt_lower = prompt.lower()
        matches = 0
        for keyword in keywords:
            if keyword.lower() in prompt_lower:
                matches += 1
        return matches

    def _match_intent_patterns(self, prompt: str, patterns: List[str]) -> int:
        """Count regex pattern matches in prompt."""
        matches = 0
        for pattern in patterns:
            try:
                if re.search(pattern, prompt, re.IGNORECASE):
                    matches += 1
            except re.error:
                continue
        return matches

    def _match_file_patterns(self, file_paths: List[str], path_patterns: List[str]) -> bool:
        """Check if any file paths match the patterns."""
        if not file_paths or not path_patterns:
            return False

        for file_path in file_paths:
            for pattern in path_patterns:
                # Convert glob pattern to regex
                regex_pattern = pattern.replace('**/', '.*').replace('*', '[^/]*').replace('?', '.')
                if re.search(regex_pattern, file_path):
                    return True
        return False

    def _check_file_content_patterns(self, file_paths: List[str], content_patterns: List[str]) -> bool:
        """Check if file contents match any patterns."""
        if not file_paths or not content_patterns:
            return False

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for pattern in content_patterns:
                        try:
                            if re.search(pattern, content, re.MULTILINE):
                                return True
                        except re.error:
                            continue
            except:
                continue

        return False

    def analyze_prompt(self, prompt: str, file_paths: Optional[List[str]] = None) -> List[Dict]:
        """
        Analyze user prompt and file context to suggest relevant skills.

        Returns list of skill suggestions with scores and reasons.
        """
        if file_paths is None:
            file_paths = []

        suggestions = []
        skills = self.rules.get("skills", {})

        for skill_name, skill_config in skills.items():
            # Skip metadata sections
            if skill_name in ["notes", "metadata"]:
                continue

            # Check skip conditions first
            if self._check_skip_conditions(skill_name, skill_config, file_paths):
                continue

            score = 0
            reasons = []

            # Check prompt triggers
            prompt_triggers = skill_config.get("promptTriggers", {})

            # Keyword matching
            keywords = prompt_triggers.get("keywords", [])
            if keywords:
                keyword_matches = self._match_keywords(prompt, keywords)
                if keyword_matches > 0:
                    score += keyword_matches * 2
                    reasons.append(f"Matched {keyword_matches} keyword(s)")

            # Intent pattern matching
            intent_patterns = prompt_triggers.get("intentPatterns", [])
            if intent_patterns:
                intent_matches = self._match_intent_patterns(prompt, intent_patterns)
                if intent_matches > 0:
                    score += intent_matches * 5
                    reasons.append(f"Matched {intent_matches} intent pattern(s)")

            # File triggers
            file_triggers = skill_config.get("fileTriggers", {})

            # Path pattern matching
            path_patterns = file_triggers.get("pathPatterns", [])
            if path_patterns and self._match_file_patterns(file_paths, path_patterns):
                score += 10
                reasons.append("Working on relevant files")

            # Content pattern matching
            content_patterns = file_triggers.get("contentPatterns", [])
            if content_patterns and self._check_file_content_patterns(file_paths, content_patterns):
                score += 15
                reasons.append("File content matches patterns")

            # Only suggest if score is above threshold
            if score >= 2:
                priority = skill_config.get("priority", "medium")
                priority_weight = {
                    "critical": 100,
                    "high": 50,
                    "medium": 25,
                    "low": 10
                }.get(priority, 25)

                final_score = score + priority_weight

                enforcement = skill_config.get("enforcement", "suggestion")
                # Get the appropriate message based on enforcement type
                if enforcement == "block":
                    message = skill_config.get("blockMessage", skill_config.get("suggestionMessage", ""))
                else:
                    message = skill_config.get("suggestionMessage", "")

                suggestions.append({
                    "skill": skill_name,
                    "score": final_score,
                    "reasons": reasons,
                    "description": skill_config.get("description", ""),
                    "message": message,
                    "enforcement": enforcement,
                    "priority": priority
                })

        # Sort by score (highest first)
        suggestions.sort(key=lambda x: x["score"], reverse=True)

        return suggestions

    def format_suggestion(self, suggestion: Dict) -> str:
        """Format a skill suggestion for display."""
        skill_name = suggestion["skill"]
        enforcement = suggestion["enforcement"]

        # Different formatting based on enforcement
        if enforcement == "block":
            header = f"🚫 REQUIRED: {skill_name}"
        elif enforcement == "warn":
            header = f"⚠️  WARNING: Consider using {skill_name}"
        else:
            header = f"💡 Suggestion: {skill_name}"

        output = [f"\n{header}"]
        output.append(f"   {suggestion['description']}")

        if suggestion.get("message"):
            output.append(f"\n{suggestion['message']}")

        output.append(f"\n   Use: /invoke {skill_name}")

        return "\n".join(output)


def main():
    """Main entry point for the skill suggester."""
    if len(sys.argv) < 2:
        print("Usage: skill-suggester.py <prompt> [file_paths...]", file=sys.stderr)
        sys.exit(1)

    prompt = sys.argv[1]
    file_paths = sys.argv[2:] if len(sys.argv) > 2 else []

    # Find the rules file
    script_dir = Path(__file__).parent
    rules_path = script_dir.parent / "skills" / "skill-rules.json"

    if not rules_path.exists():
        # Try alternative location
        rules_path = Path.cwd() / ".claude" / "skills" / "skill-rules.json"

    if not rules_path.exists():
        print(f"Error: Could not find skill-rules.json at {rules_path}", file=sys.stderr)
        sys.exit(1)

    suggester = SkillSuggester(str(rules_path))
    suggestions = suggester.analyze_prompt(prompt, file_paths)

    # Output suggestions
    if suggestions:
        # Show top suggestion(s)
        top_suggestion = suggestions[0]

        # Only show if score is significant
        if top_suggestion["score"] >= 30:
            print(suggester.format_suggestion(top_suggestion))

            # Show second suggestion if it's also high score
            if len(suggestions) > 1 and suggestions[1]["score"] >= 50:
                print(suggester.format_suggestion(suggestions[1]))


if __name__ == "__main__":
    main()
