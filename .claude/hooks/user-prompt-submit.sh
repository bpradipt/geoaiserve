#!/usr/bin/env bash
#
# UserPromptSubmit Hook for Claude Code
# Automatically suggests relevant skills based on user prompts and file context
#
# This hook runs before each user prompt is submitted to Claude.
# It analyzes the prompt and current working context to suggest relevant skills.
#
# Environment Variables Available:
#   USER_PROMPT          - The user's prompt text
#   WORKING_DIRECTORY    - Current working directory
#   OPEN_FILES           - List of currently open files (newline-separated)
#   CLAUDE_SESSION_ID    - Unique session identifier
#
# Return Codes:
#   0 - Continue normally (allow prompt)
#   1 - Block prompt (only if skill enforcement is "block")

set -euo pipefail

# Configuration
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SUGGESTER="${HOOK_DIR}/skill-suggester.py"
ENABLE_SUGGESTIONS="${CLAUDE_SKILL_SUGGESTIONS:-true}"

# Color codes for terminal output
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Function to log debug messages
debug() {
    if [[ "${CLAUDE_DEBUG:-false}" == "true" ]]; then
        echo "[DEBUG] $*" >&2
    fi
}

# Function to check if skill suggestions are enabled
is_enabled() {
    [[ "$ENABLE_SUGGESTIONS" == "true" ]]
}

# Main hook logic
main() {
    # Check if suggestions are enabled
    if ! is_enabled; then
        debug "Skill suggestions disabled via CLAUDE_SKILL_SUGGESTIONS"
        exit 0
    fi

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        debug "Python3 not found, skipping skill suggestions"
        exit 0
    fi

    # Check if suggester script exists
    if [[ ! -f "$SKILL_SUGGESTER" ]]; then
        debug "Skill suggester not found at: $SKILL_SUGGESTER"
        exit 0
    fi

    # Get the user prompt
    PROMPT="${USER_PROMPT:-}"
    if [[ -z "$PROMPT" ]]; then
        debug "No user prompt provided"
        exit 0
    fi

    # Get working directory and open files
    WORK_DIR="${WORKING_DIRECTORY:-$(pwd)}"
    FILES="${OPEN_FILES:-}"

    debug "Analyzing prompt: ${PROMPT:0:50}..."
    debug "Working directory: $WORK_DIR"

    # Prepare file paths array
    FILE_ARGS=()
    if [[ -n "$FILES" ]]; then
        while IFS= read -r file; do
            if [[ -n "$file" ]]; then
                FILE_ARGS+=("$file")
            fi
        done <<< "$FILES"
    fi

    # Alternative: Try to detect Python files in current directory
    if [[ ${#FILE_ARGS[@]} -eq 0 ]]; then
        # Check if we're in a Python project (has .py files or pyproject.toml)
        if [[ -f "pyproject.toml" ]] || [[ -f "requirements.txt" ]] || compgen -G "*.py" > /dev/null 2>&1; then
            # Add common Python project files for context
            [[ -f "main.py" ]] && FILE_ARGS+=("main.py")
            [[ -f "app.py" ]] && FILE_ARGS+=("app.py")
            [[ -f "pyproject.toml" ]] && FILE_ARGS+=("pyproject.toml")
        fi
    fi

    # Run the skill suggester
    debug "Running skill suggester with ${#FILE_ARGS[@]} file(s)"

    SUGGESTION_OUTPUT=""
    if [[ ${#FILE_ARGS[@]} -gt 0 ]]; then
        SUGGESTION_OUTPUT=$(python3 "$SKILL_SUGGESTER" "$PROMPT" "${FILE_ARGS[@]}" 2>/dev/null || true)
    else
        SUGGESTION_OUTPUT=$(python3 "$SKILL_SUGGESTER" "$PROMPT" 2>/dev/null || true)
    fi

    # Display suggestions if any
    if [[ -n "$SUGGESTION_OUTPUT" ]]; then
        echo ""
        echo "$SUGGESTION_OUTPUT"
        echo ""

        # Check if this is a blocking suggestion (contains "REQUIRED")
        if echo "$SUGGESTION_OUTPUT" | grep -q "🚫 REQUIRED"; then
            echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${RED}⛔ BLOCKED: Skill is required before proceeding${NC}"
            echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo ""
            echo -e "${YELLOW}To bypass this check:${NC}"
            echo -e "  1. Invoke the required skill using the command above"
            echo -e "  2. Add skip marker to your file (see message above)"
            echo -e "  3. Set environment variable to skip (see message above)"
            echo ""
            # Block the prompt
            exit 1
        fi
    fi

    # Allow the prompt to continue
    exit 0
}

# Run main function
main "$@"
